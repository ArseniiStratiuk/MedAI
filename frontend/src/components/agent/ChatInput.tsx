"use client";

import React, { useState, useRef, useCallback, KeyboardEvent } from "react";
import { cn } from "@/lib/utils";
import { apiClient } from "@/lib/api/client";
import {
  Send,
  Paperclip,
  Image as ImageIcon,
  AudioLines,
  FileText,
  X,
  Mic,
  Square,
} from "lucide-react";

interface ChatInputProps {
  onSend: (message: string, attachments?: File[]) => void;
  disabled?: boolean;
  className?: string;
  value?: string;
  onValueChange?: (value: string) => void;
}

/** Encode a Float32Array as a 16-bit PCM WAV file (ArrayBuffer). */
function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(view, 8, "WAVE");

  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);          // chunk size
  view.setUint16(20, 1, true);           // PCM
  view.setUint16(22, 1, true);           // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);           // block align
  view.setUint16(34, 16, true);          // bits per sample

  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, numSamples * 2, true);

  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export function ChatInput({ onSend, disabled, className, value: externalValue, onValueChange }: ChatInputProps) {

  const [text, setText] = useState("");
  const isControlled = externalValue !== undefined;
  const currentText = isControlled ? externalValue : text;
  const [files, setFiles] = useState<File[]>([]);
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const handleSend = () => {
    if (!currentText.trim() && files.length === 0) return;
    onSend(currentText.trim(), files.length > 0 ? files : undefined);
    if (isControlled) {
      onValueChange?.("");
    } else {
      setText("");
    }
    setFiles([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  };

  const handleFileSelect = (accept: string) => {
    if (fileInputRef.current) {
      fileInputRef.current.accept = accept;
      fileInputRef.current.click();
    }
    setShowAttachMenu(false);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith("image/")) return ImageIcon;
    if (file.type.startsWith("audio/")) return AudioLines;
    return FileText;
  };

  // ── Voice recording ────────────────────────────────────

  const updateText = useCallback(
    (transcription: string) => {
      if (isControlled) {
        const current = externalValue ?? "";
        const sep = current.trim() ? " " : "";
        onValueChange?.(current + sep + transcription);
      } else {
        setText((prev) => {
          const sep = prev.trim() ? " " : "";
          return prev + sep + transcription;
        });
      }
      // Auto-resize textarea
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.style.height = "auto";
          textareaRef.current.style.height =
            Math.min(textareaRef.current.scrollHeight, 160) + "px";
        }
      }, 0);
    },
    [isControlled, externalValue, onValueChange]
  );

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Prefer audio/webm, fall back to default mimeType
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : undefined;
      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks to release the mic
        stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType });
        if (blob.size === 0) return;

        setIsTranscribing(true);
        try {
          // Convert recorded audio → WAV using AudioContext for Whisper compatibility
          const arrayBuf = await blob.arrayBuffer();
          const audioCtx = new AudioContext({ sampleRate: 16000 });
          const decoded = await audioCtx.decodeAudioData(arrayBuf);
          const pcm = decoded.getChannelData(0); // mono float32

          // Encode as 16-bit WAV
          const wavBytes = encodeWav(pcm, 16000);

          // Convert to base64 in chunks (spread operator crashes on large arrays)
          const bytes = new Uint8Array(wavBytes);
          const chunkSize = 8192;
          let binary = "";
          for (let i = 0; i < bytes.length; i += chunkSize) {
            binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
          }
          const base64 = btoa(binary);

          const result = await apiClient.transcribe(base64);
          if (result.error) {
            console.error("Transcription error:", result.error);
          } else if (result.transcription) {
            updateText(result.transcription);
          }
          await audioCtx.close();
        } catch (err) {
          console.error("Transcription failed:", err);
          alert("Voice transcription failed. Please try again.");
        } finally {
          setIsTranscribing(false);
        }
      };

      mediaRecorder.start(250); // collect data every 250ms for reliability
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    } catch (err) {
      console.error("Mic access denied:", err);
      alert("Microphone access was denied. Please allow microphone access and try again.");
    }
  }, [updateText]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    setIsRecording(false);
  }, []);

  const handleMicToggle = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  return (
    <div className={cn("relative", className)}>
      {/* File previews */}
      {files.length > 0 && (
        <div className="flex gap-2 px-4 pb-2 flex-wrap">
          {files.map((file, i) => {
            const Icon = getFileIcon(file);
            return (
              <div
                key={i}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-brand-50 dark:bg-brand-900/20 border border-brand-200 dark:border-brand-800 text-sm"
              >
                <Icon size={14} className="text-brand-500" />
                <span className="text-xs text-gray-700 dark:text-gray-300 max-w-[120px] truncate">
                  {file.name}
                </span>
                <button
                  onClick={() => removeFile(i)}
                  className="p-0.5 rounded hover:bg-brand-100 dark:hover:bg-brand-800 transition"
                >
                  <X size={12} className="text-gray-400" />
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* Input bar */}
      <div
        className={cn(
          "flex items-end gap-2 px-4 py-3",
          "bg-white dark:bg-surface-dark-2 border border-gray-200 dark:border-gray-700",
          "rounded-2xl transition-all duration-200",
          "focus-within:ring-2 focus-within:ring-brand-500/20 focus-within:border-brand-400 dark:focus-within:border-brand-600",
          "neo-shadow"
        )}
      >
        {/* Attach button */}
        <div className="relative">
          <button
            onClick={() => setShowAttachMenu(!showAttachMenu)}
            className="p-2 rounded-xl text-gray-400 hover:text-brand-500 hover:bg-brand-50 dark:hover:bg-brand-900/20 transition"
            aria-label="Attach file"
            aria-expanded={showAttachMenu}
            aria-haspopup="menu"
          >
            <Paperclip size={18} />
          </button>

          {showAttachMenu && (
            <div role="menu" className="absolute bottom-full left-0 mb-2 py-2 w-48 bg-white dark:bg-surface-dark-2 rounded-xl border border-gray-200 dark:border-gray-700 neo-shadow-lg z-50">
              <button
                onClick={() => handleFileSelect("image/*")}
                role="menuitem"
                className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-dark-3 transition"
              >
                <ImageIcon size={16} className="text-accent-cyan" />
                Medical Image
              </button>
              <button
                onClick={() => handleFileSelect("audio/*")}
                role="menuitem"
                className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-dark-3 transition"
              >
                <AudioLines size={16} className="text-accent-amber" />
                Audio Recording
              </button>
              <button
                onClick={() => handleFileSelect(".pdf,.txt,.doc,.docx")}
                role="menuitem"
                className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-dark-3 transition"
              >
                <FileText size={16} className="text-accent-violet" />
                Document / Report
              </button>
            </div>
          )}
        </div>

        {/* Textarea */}
        <label htmlFor="chat-input" className="sr-only">Clinical case description</label>
        <textarea
          id="chat-input"
          ref={textareaRef}
          value={currentText}
          onChange={(e) => {
            if (isControlled) {
              onValueChange?.(e.target.value);
            } else {
              setText(e.target.value);
            }
          }}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Describe the clinical case or ask a question..."
          disabled={disabled}
          rows={1}
          className={cn(
            "flex-1 resize-none bg-transparent border-none outline-none",
            "text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500",
            "leading-relaxed py-1",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        />

        {/* Voice button */}
        <button
          onClick={handleMicToggle}
          disabled={disabled || isTranscribing}
          className={cn(
            "p-2 rounded-xl transition-all duration-200",
            isRecording
              ? "text-red-500 bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/30 animate-pulse"
              : isTranscribing
                ? "text-amber-500 bg-amber-50 dark:bg-amber-900/20 opacity-70 cursor-wait"
                : "text-gray-400 hover:text-accent-amber hover:bg-amber-50 dark:hover:bg-amber-900/20"
          )}
          aria-label={isRecording ? "Stop recording" : isTranscribing ? "Transcribing…" : "Start voice input"}
          title={isRecording ? "Click to stop recording" : isTranscribing ? "Transcribing with Whisper…" : "Voice input"}
        >
          {isRecording ? <Square size={18} /> : <Mic size={18} />}
        </button>

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={disabled || (!currentText.trim() && files.length === 0)}
          aria-label="Send message"
          className={cn(
            "p-2.5 rounded-xl transition-all duration-200",
            currentText.trim() || files.length > 0
              ? "bg-brand-500 hover:bg-brand-600 text-white shadow-lg shadow-brand-500/25"
              : "bg-gray-100 dark:bg-surface-dark-3 text-gray-400 cursor-not-allowed"
          )}
        >
          <Send size={16} />
        </button>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileChange}
        aria-label="Upload file"
        multiple
      />
    </div>
  );
}
