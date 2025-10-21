"use client";
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react"; // Added useCallback
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { CircleStop, Mic, Play, Pause, Trash, Loader2, Upload } from "lucide-react"; // Added Upload icon
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";

// --- ffmpeg.wasm imports ---
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
// --- End ffmpeg.wasm imports ---

type Props = {
  className?: string;
  timerClassName?: string;
};

let timerTimeout: NodeJS.Timeout;

const padWithLeadingZeros = (num: number, length: number): string => {
  return String(num).padStart(length, "0");
};

// *** MODIFICATION: Added loading_ffmpeg, converting phases ***
type RecordingPhase = "idle" | "loading_ffmpeg" | "recording" | "converting" | "review";

const CLIP_DURATION_MS = 5000; // 5 seconds per clip

export const AudioRecorderWithVisualizer = ({ className, timerClassName }: Props) => {
  const { theme } = useTheme();
  // States
  const [recordingPhase, setRecordingPhase] = useState<RecordingPhase>("idle");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [timer, setTimer] = useState<number>(0);
  const [finalAudioBlob, setFinalAudioBlob] = useState<Blob | null>(null);
  const [isPlayingBack, setIsPlayingBack] = useState<boolean>(false);
  const [currentPlaybackTime, setCurrentPlaybackTime] = useState<number>(0);
  const [totalAudioDuration, setTotalAudioDuration] = useState<number>(0);
  // *** NEW ffmpeg.wasm states ***
  const [ffmpegLoaded, setFfmpegLoaded] = useState<boolean>(false);
  const [ffmpegLoadingMessage, setFfmpegLoadingMessage] = useState<string>("");
  // *** End NEW ffmpeg.wasm states ***

  // Refs for the primary MediaStream and its AudioContext/Analyser for LIVE visualization
  const mediaRecorderRef = useRef<{
    stream: MediaStream | null;
    analyser: AnalyserNode | null;
    audioContext: AudioContext | null;
    mediaRecorder: MediaRecorder | null; // This is now solely for the frontend's full recording
  }>({
    stream: null,
    analyser: null,
    audioContext: null,
    mediaRecorder: null, // Initialized for the frontend's full recording
  });

  // *** REMOVED: Refs for backend continuous clipping (no longer needed) ***
  const backendClipRecorderRef = useRef<MediaRecorder | null>(null);
  const backendClipChunks = useRef<BlobPart[]>([]);
  const backendRecorderTimeoutId = useRef<NodeJS.Timeout | null>(null);

  // Ref to store history of all recorded chunks for frontend full playback
  const allRecordedChunks = useRef<BlobPart[]>([]);

  // *** NEW: ffmpeg.wasm instance ref ***
  const ffmpegRef = useRef<FFmpeg | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null); // For playback
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const liveAnimationRef = useRef<number | null>(null); // For live recording waveform
  const reviewAnimationRef = useRef<number | null>(null); // For review waveform

  // Refs for review/playback analysis
  const playbackAudioContextRef = useRef<AudioContext | null>(null);
  const playbackSourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const playbackAnalyserRef = useRef<AnalyserNode | null>(null);
  const reviewAudioBufferRef = useRef<AudioBuffer | null>(null); // To store decoded audio buffer for review
  const fileInputRef = useRef<HTMLInputElement>(null);

  const latestRecordingPhase = useRef<RecordingPhase>("idle"); // To store the latest recordingPhase

  // Sync latestRecordingPhase ref with recordingPhase state
  useEffect(() => {
    latestRecordingPhase.current = recordingPhase;
  }, [recordingPhase]);

  const loadFFmpeg = useCallback(async () => {
    if (ffmpegRef.current) {
      setFfmpegLoaded(true); // Already loaded
      return;
    }
    setRecordingPhase("loading_ffmpeg");
    setFfmpegLoadingMessage("Loading FFmpeg core...");
    try {
      const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd"; // Use a stable CDN URL
      const ffmpeg = new FFmpeg();
      ffmpeg.on("log", ({ message }) => {
        // console.log("[ffmpeg.wasm log]", message); // Uncomment for verbose ffmpeg logs
        setFfmpegLoadingMessage(message);
      });
      await ffmpeg.load({
        coreURL: `${baseURL}/ffmpeg-core.js`,
        wasmURL: `${baseURL}/ffmpeg-core.wasm`,
        workerURL: `${baseURL}/ffmpeg-core.worker.js`,
      });
      ffmpegRef.current = ffmpeg;
      setFfmpegLoaded(true);
      setRecordingPhase("idle"); // Back to idle after loading
      console.log("FFmpeg loaded successfully!");
    } catch (e) {
      console.error("Failed to load FFmpeg:", e);
      setFfmpegLoadingMessage("Failed to load FFmpeg. Check console for details.");
      setFfmpegLoaded(false);
      setRecordingPhase("idle"); // Back to idle on error
    }
  }, []); // Empty dependency array means it's created once

  // Effect to load FFmpeg when component mounts
  useEffect(() => {
    loadFFmpeg();
  }, [loadFFmpeg]);

  // Calculate the hours, minutes, and seconds from the timer
  const hours = Math.floor(timer / 3600);
  const minutes = Math.floor((timer % 3600) / 60);
  const seconds = timer % 60;

  // Split the hours, minutes, and seconds into individual digits
  const [hourLeft, hourRight] = useMemo(() => padWithLeadingZeros(hours, 2).split(""), [hours]);
  const [minuteLeft, minuteRight] = useMemo(
    () => padWithLeadingZeros(minutes, 2).split(""),
    [minutes]
  );
  const [secondLeft, secondRight] = useMemo(
    () => padWithLeadingZeros(seconds, 2).split(""),
    [seconds]
  );
  // --- Backend Recorder Cycle Management ---
  const _startBackendRecordingCycle = (stream: MediaStream) => {
    _stopBackendRecordingCycle(); // Clean up any previous cycle's timeout or recorder instance

    if (!stream.active) {
      console.warn("Attempted to start backend recording cycle with an inactive stream. Skipping.");
      return;
    }
    if (!ffmpegRef.current || !ffmpegLoaded) {
      console.error("FFmpeg not loaded for backend clip conversion. Cannot start backend cycle.");
      return;
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/mpeg";
    const options = { mimeType };

    backendClipRecorderRef.current = new MediaRecorder(stream, options);
    backendClipChunks.current = []; // Reset chunks for the new clip

    backendClipRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) {
        backendClipChunks.current.push(e.data);
      }
    };

    backendClipRecorderRef.current.onstop = async () => {
      console.log(
        `Backend clip recorder stopped. Chunks collected: ${backendClipChunks.current.length}`
      );
      if (backendClipChunks.current.length > 0) {
        const ffmpeg = ffmpegRef.current!; // Assert non-null, checked above
        const fullWebmClipBlob = new Blob(backendClipChunks.current, { type: mimeType });

        try {
          const inputFileName = `clip_input_${Date.now()}.webm`;
          const outputFileName = `clip_output_${Date.now()}.wav`;

          // *** NEW: Convert the small WebM clip to WAV using ffmpeg.wasm ***
          await ffmpeg.writeFile(inputFileName, await fetchFile(fullWebmClipBlob));
          await ffmpeg.exec(["-i", inputFileName, outputFileName]);
          const data = await ffmpeg.readFile(outputFileName);
          const wavClipBlob = new Blob([(data as any).buffer], { type: "audio/wav" });

          await sendAudioToBackend(wavClipBlob); // Send the converted WAV clip to backend
          console.log(`Backend clip (WAV) sent after ffmpeg.wasm conversion.`);

          // Clean up files in FFmpeg's virtual file system
          await ffmpeg.deleteFile(inputFileName);
          await ffmpeg.deleteFile(outputFileName);
        } catch (e) {
          console.error("Error converting/sending backend clip with ffmpeg.wasm:", e);
        }
      }
      // Immediately restart the cycle for the next clip if still in recording phase
      if (
        latestRecordingPhase.current === "recording" &&
        mediaRecorderRef.current.stream &&
        mediaRecorderRef.current.stream.active
      ) {
        console.log("Restarting backend recording cycle...");
        _startBackendRecordingCycle(stream); // Recursive call to start the next clip with the same stream
      } else {
        console.log(
          "Not restarting backend recording cycle, recordingPhase is not 'recording' or stream inactive."
        );
      }
    };

    backendClipRecorderRef.current.onerror = (event) => {
      console.error("Backend MediaRecorder error:", event);
      if (
        latestRecordingPhase.current === "recording" &&
        mediaRecorderRef.current.stream &&
        mediaRecorderRef.current.stream.active
      ) {
        console.log("Restarting backend recording cycle due to error...");
        _startBackendRecordingCycle(stream);
      }
    };

    backendClipRecorderRef.current.start();
    console.log(`Backend clip recorder started for ${CLIP_DURATION_MS / 1000} seconds.`);

    backendRecorderTimeoutId.current = setTimeout(() => {
      if (backendClipRecorderRef.current && backendClipRecorderRef.current.state === "recording") {
        backendClipRecorderRef.current.stop();
      }
    }, CLIP_DURATION_MS);
  };

  const _stopBackendRecordingCycle = () => {
    if (backendRecorderTimeoutId.current) {
      clearTimeout(backendRecorderTimeoutId.current);
      backendRecorderTimeoutId.current = null;
    }
    if (backendClipRecorderRef.current && backendClipRecorderRef.current.state === "recording") {
      backendClipRecorderRef.current.stop();
    }
    if (backendClipRecorderRef.current) {
      backendClipRecorderRef.current.onstop = null;
      backendClipRecorderRef.current = null;
    }
    backendClipChunks.current = [];
    console.log("Backend recording cycle explicitly stopped and cleaned up.");
  };
  // --- End Backend Recorder Cycle Management ---
  const sendAudioToBackend = async (audioBlob: Blob) => {
    const formData = new FormData();
    // *** MODIFICATION: Now expects WAV, not WebM, from frontend ffmpeg.wasm ***
    formData.append("audio", audioBlob, `audio_final_${Date.now()}.wav`);

    try {
      console.log(
        `Attempting to send FINAL WAV clip (${(audioBlob.size / 1024).toFixed(
          2
        )} KB) to /api/audio-upload`
      );
      const response = await fetch("/api/audio-upload", {
        method: "POST",
        body: formData,
      });

      console.log(
        "Response received for FINAL WAV upload. Status:",
        response.status,
        "URL:",
        response.url
      );

      if (response.ok) {
        console.log("FINAL WAV clip sent successfully! Backend response status:", response.status);
        try {
          const responseData = await response.json();
          console.log("Backend JSON response data:", responseData);
        } catch (jsonError) {
          console.warn(
            "Could not parse JSON response from backend (might be empty or non-JSON success).",
            jsonError
          );
          const textResponse = await response.text();
          console.log(
            "Backend response as text (first 200 chars):",
            textResponse.substring(0, 200) + "..."
          );
        }
      } else {
        const errorText = await response.text();
        let errorData = { message: errorText };

        try {
          const jsonError = JSON.parse(errorText);
          errorData = jsonError;
        } catch (e) {}

        console.error(
          "Failed to send FINAL WAV clip:",
          response.status,
          response.statusText,
          errorData
        );
      }
    } catch (error) {
      console.error("Error sending FINAL WAV clip (network or CORS issue):", error);
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const canvasCtx = canvas.getContext("2d");
      if (canvasCtx) {
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  // Drawing utilities for the canvas (moved outside useEffect for access)
  const drawLiveWaveform = (
    canvasCtx: CanvasRenderingContext2D,
    WIDTH: number,
    HEIGHT: number,
    dataArray: Uint8Array
  ) => {
    canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);
    canvasCtx.fillStyle = "#939393";

    const barWidth = 1;
    const spacing = 1;
    const maxBarHeight = HEIGHT / 1.2;
    const numBars = Math.floor(WIDTH / (barWidth + spacing));

    for (let i = 0; i < numBars; i++) {
      const dataIndex = Math.floor((i * dataArray.length) / numBars);
      const barHeight = Math.pow(dataArray[dataIndex] / 128.0, 8) * maxBarHeight;
      const x = (barWidth + spacing) * i;
      const y = HEIGHT / 2 - barHeight / 2;
      canvasCtx.fillRect(x, y, barWidth, barHeight);
    }
  };

  const drawReviewWaveform = (
    canvasCtx: CanvasRenderingContext2D,
    WIDTH: number,
    HEIGHT: number,
    audioBuffer: AudioBuffer,
    currentPlaybackTime: number = 0,
    totalAudioDuration: number = 0
  ) => {
    canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);
    canvasCtx.fillStyle = "#939393"; // *** TEMPORARY: Use a very distinct color to force visibility ***

    const barWidth = 2;
    const barSpacing = 1;
    const maxBarHeight = HEIGHT / 2.5;
    const numBars = Math.floor(WIDTH / (barWidth + barSpacing));

    const channelData = audioBuffer.getChannelData(0); // Use the first channel for waveform data

    for (let i = 0; i < numBars; i++) {
      const dataIndex = Math.floor(i * (channelData.length / numBars));
      const amplitude = Math.abs(channelData[dataIndex]);

      // *** MODIFICATION: More aggressive amplitude scaling for visibility ***
      // Scale amplitude (0.0-1.0) to a more visually impactful range.
      // Multiply by a factor (e.g., 5) to ensure even small amplitudes get boosted.
      // Then apply power to shape, or just use linear scaling for initial test.
      const scaledAmplitude = amplitude * 5; // Boost amplitude considerably
      const barHeight = Math.min(HEIGHT, Math.max(0, scaledAmplitude * maxBarHeight)); // Clamp to canvas height, no power for now

      const x = (barWidth + barSpacing) * i;
      const y = HEIGHT / 2 - barHeight / 2;

      canvasCtx.fillRect(x, y, barWidth, barHeight);
    }

    // --- Draw Playback Head (vertical line) ---
    if (currentPlaybackTime > 0 && totalAudioDuration > 0) {
      const playbackPositionX = (currentPlaybackTime / totalAudioDuration) * WIDTH;
      canvasCtx.beginPath();
      canvasCtx.strokeStyle = "red";
      canvasCtx.lineWidth = 2;
      canvasCtx.lineCap = "round";
      canvasCtx.moveTo(playbackPositionX, 0);
      canvasCtx.lineTo(playbackPositionX, HEIGHT);
      canvasCtx.stroke();
    }
  };

  // --- Backend Recorder Cycle Management ---

  // --- End Backend Recorder Cycle Management ---

  function startRecording() {
    if (!ffmpegLoaded) {
      alert("FFmpeg is still loading or failed to load. Please wait or refresh.");
      return;
    }
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          setRecordingPhase("recording");
          setIsRecording(true);
          setTimer(0);
          allRecordedChunks.current = [];
          setFinalAudioBlob(null); // Clear any previous stitched audio
          console.log("startRecording: Initializing for new recording.");

          const AudioContext = window.AudioContext;
          const audioCtx = new AudioContext();
          const analyser = audioCtx.createAnalyser();
          const source = audioCtx.createMediaStreamSource(stream);
          source.connect(analyser);

          mediaRecorderRef.current = {
            stream,
            analyser,
            audioContext: audioCtx,
            mediaRecorder: null, // Will be set below
          };

          const mimeType = MediaRecorder.isTypeSupported("audio/webm")
            ? "audio/webm"
            : MediaRecorder.isTypeSupported("audio/mpeg")
            ? "audio/mpeg"
            : "audio/wav";

          if (!MediaRecorder.isTypeSupported(mimeType)) {
            console.error(
              `CRITICAL: Browser does not support any known audio formats for MediaRecorder.`
            );
            alert("Your browser does not support any known audio formats for recording.");
            setRecordingPhase("idle");
            setIsRecording(false);
            return;
          }
          console.log(`MediaRecorder for full recording will use MIME type: ${mimeType}`);

          // --- Full Recording for Frontend Playback/Conversion ---
          const fullMediaRecorder = new MediaRecorder(stream, { mimeType });
          const currentAllRecordedChunksRef = allRecordedChunks;
          fullMediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
              currentAllRecordedChunksRef.current.push(e.data);
              // console.log(
              //   `Frontend full recorder collected chunk: ${e.data.size} bytes. Total chunks: ${currentAllRecordedChunksRef.current.length}`
              // );
            }
          };
          fullMediaRecorder.onstop = () => {
            console.log("Frontend full MediaRecorder stopped.");
          };
          fullMediaRecorder.start(200); // Start with a 200ms timeslice to populate chunks regularly
          mediaRecorderRef.current.mediaRecorder = fullMediaRecorder;

          // --- Start Backend Recording Cycle ---
          _startBackendRecordingCycle(stream); // Start the continuous clipping for backend
        })
        .catch((error) => {
          alert("Error accessing microphone: " + error.message);
          console.error("Error accessing microphone:", error);
          setRecordingPhase("idle");
          setIsRecording(false);
        });
    } else {
      alert("getUserMedia not supported on your browser!");
      setRecordingPhase("idle");
      setIsRecording(false);
    }
  }

  async function stopListening() {
    const { stream, analyser, audioContext, mediaRecorder } = mediaRecorderRef.current;

    // --- Stop Backend Recording Cycle ---
    _stopBackendRecordingCycle(); // Stop the continuous backend clipping

    // Stop the full recording MediaRecorder
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop(); // This recorder collects all chunks for frontend playback
    }
    // Stop actual MediaStream tracks
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    // Disconnect analyser and close audio context
    if (analyser) {
      analyser.disconnect();
    }
    if (audioContext) {
      audioContext.close();
    }

    // Clear live animation frame
    if (liveAnimationRef.current) {
      cancelAnimationFrame(liveAnimationRef.current);
    }

    setIsRecording(false); // Stop live visualization flag
    clearTimeout(timerTimeout); // Stop the timer

    console.log("stopListening: Processing full recorded audio for frontend playback.");

    if (allRecordedChunks.current.length > 0) {
      setRecordingPhase("converting"); // New phase: show conversion progress
      setFfmpegLoadingMessage("Converting full audio (WebM to WAV) for playback...");

      try {
        const firstChunk = allRecordedChunks.current[0];
        const mimeType = firstChunk instanceof Blob ? firstChunk.type : "audio/webm";
        const fullWebmBlob = new Blob(allRecordedChunks.current, { type: mimeType });

        if (!ffmpegRef.current || !ffmpegLoaded) {
          throw new Error("FFmpeg not loaded for conversion.");
        }

        const ffmpeg = ffmpegRef.current;
        const inputFileName = `full_input_${Date.now()}.webm`;
        const outputFileName = `full_output_${Date.now()}.wav`;

        await ffmpeg.writeFile(inputFileName, await fetchFile(fullWebmBlob));
        console.log(`FFmpeg: Wrote ${inputFileName} to virtual file system.`);
        setFfmpegLoadingMessage("Running FFmpeg conversion for full playback...");

        await ffmpeg.exec(["-i", inputFileName, outputFileName]);
        console.log(`FFmpeg: Conversion complete. Reading ${outputFileName}.`);
        setFfmpegLoadingMessage("Reading converted WAV for playback...");

        const data = await ffmpeg.readFile(outputFileName);
        const wavBlob = new Blob([(data as any).buffer], { type: "audio/wav" });
        console.log(`FFmpeg: Converted full WAV blob created (size: ${wavBlob.size} bytes).`);

        setFinalAudioBlob(wavBlob); // Set this for frontend playback

        const tempAudioContext = new window.AudioContext();
        const audioBuffer = await tempAudioContext.decodeAudioData(await wavBlob.arrayBuffer());
        reviewAudioBufferRef.current = audioBuffer;
        setTotalAudioDuration(audioBuffer.duration);
        tempAudioContext.close();
        setRecordingPhase("review");
        console.log(
          "stopListening: AudioBuffer decoded from converted WAV, transitioning to review."
        );

        // Clean up full recording files in FFmpeg's virtual file system
        await ffmpeg.deleteFile(inputFileName);
        await ffmpeg.deleteFile(outputFileName);
      } catch (e) {
        console.error(
          "stopListening: Error during FFmpeg conversion or decoding for full playback:",
          e
        );
        alert("Audio processing failed for playback: " + (e as Error).message);
        setRecordingPhase("idle");
        setTotalAudioDuration(0);
        setFinalAudioBlob(null);
        reviewAudioBufferRef.current = null;
      }
    } else {
      console.warn("stopListening: No audio chunks recorded for frontend review.");
      setRecordingPhase("idle");
      setTotalAudioDuration(0);
      setFinalAudioBlob(null);
      reviewAudioBufferRef.current = null;
    }

    mediaRecorderRef.current = {
      stream: null,
      analyser: null,
      audioContext: null,
      mediaRecorder: null, // Clear the recorder instance
    };
  }

  function discardRecording() {
    console.log("Discarding recording.");
    // Stop any ongoing playback
    if (audioRef.current) {
      audioRef.current.pause();
      if (audioRef.current.src) URL.revokeObjectURL(audioRef.current.src);
      audioRef.current.src = "";
    }
    setIsPlayingBack(false);
    setCurrentPlaybackTime(0);
    setTotalAudioDuration(0);

    // Clear any previous recorded data
    setFinalAudioBlob(null);
    allRecordedChunks.current = []; // Clear chunks for the full recording
    // frontendPlaybackClipHistoryRef.current = []; // Removed - no longer needed
    reviewAudioBufferRef.current = null;

    // Clear canvas
    clearCanvas();
    if (reviewAnimationRef.current) {
      cancelAnimationFrame(reviewAnimationRef.current);
    }
    if (playbackAudioContextRef.current) {
      playbackAudioContextRef.current.close();
      playbackAudioContextRef.current = null;
    }

    setTimer(0);
    setRecordingPhase("idle"); // Return to initial state
    // ffmpeg.wasm will usually remain loaded, but we might want to terminate/reload for cleanup if necessary
    // For now, let it persist.
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!ffmpegRef.current || !ffmpegLoaded) {
      alert("FFmpeg is still loading or failed to load. Please wait or refresh.");
      return;
    }

    setRecordingPhase("converting");
    setFfmpegLoadingMessage("Converting uploaded audio for processing...");

    try {
      const inputExt = file.name.split(".").pop() || "";
      const inputName = `upload_input_${Date.now()}.${inputExt}`;
      const outputName = `upload_output_${Date.now()}.wav`;

      await ffmpegRef.current!.writeFile(inputName, await fetchFile(file));
      await ffmpegRef.current!.exec(["-i", inputName, outputName]);
      const data = await ffmpegRef.current!.readFile(outputName);
      const wavBlob = new Blob([(data as any).buffer], { type: "audio/wav" });

      await sendAudioToBackend(wavBlob);

      setFinalAudioBlob(wavBlob);

      const tempAudioContext = new window.AudioContext();
      const audioBuffer = await tempAudioContext.decodeAudioData(await wavBlob.arrayBuffer());
      reviewAudioBufferRef.current = audioBuffer;
      setTotalAudioDuration(audioBuffer.duration);
      tempAudioContext.close();
      setRecordingPhase("review");

      await ffmpegRef.current!.deleteFile(inputName);
      await ffmpegRef.current!.deleteFile(outputName);

      e.target.value = "";
      console.log("Uploaded file processed and ready for review.");
    } catch (e) {
      console.error("Error processing uploaded file:", e);
      alert("Failed to process uploaded file: " + (e as Error).message);
      setRecordingPhase("idle");
      setTotalAudioDuration(0);
      setFinalAudioBlob(null);
      reviewAudioBufferRef.current = null;
    }
  }

  const togglePlayback = async () => {
    if (!finalAudioBlob || !reviewAudioBufferRef.current) {
      console.warn("togglePlayback: No finalAudioBlob or reviewAudioBuffer.current available.");
      return;
    }

    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.onended = () => {
        console.log("togglePlayback: Audio playback ended.");
        setIsPlayingBack(false);
        setCurrentPlaybackTime(0);
        if (reviewAnimationRef.current) {
          cancelAnimationFrame(reviewAnimationRef.current);
          reviewAnimationRef.current = null;
        }
        if (playbackSourceNodeRef.current) {
          playbackSourceNodeRef.current.stop();
          playbackSourceNodeRef.current.disconnect();
          playbackSourceNodeRef.current = null;
        }
        if (playbackAnalyserRef.current) {
          playbackAnalyserRef.current.disconnect();
          playbackAnalyserRef.current = null;
        }
        if (playbackAudioContextRef.current) {
          playbackAudioContextRef.current.close();
          playbackAudioContextRef.current = null;
        }
      };
    }

    if (!audioRef.current.src || audioRef.current.src !== URL.createObjectURL(finalAudioBlob)) {
      if (audioRef.current.src) URL.revokeObjectURL(audioRef.current.src);
      audioRef.current.src = URL.createObjectURL(finalAudioBlob);
      audioRef.current.load();
      console.log("togglePlayback: Set new audioRef.current.src and loaded.");
    }

    if (isPlayingBack) {
      console.log("togglePlayback: Pausing playback.");
      audioRef.current.pause();
      setIsPlayingBack(false);
      if (reviewAnimationRef.current) {
        cancelAnimationFrame(reviewAnimationRef.current);
        reviewAnimationRef.current = null;
      }
      if (playbackSourceNodeRef.current) {
        playbackSourceNodeRef.current.stop();
        playbackSourceNodeRef.current.disconnect();
        playbackSourceNodeRef.current = null;
      }
      if (playbackAnalyserRef.current) {
        playbackAnalyserRef.current.disconnect();
        playbackAnalyserRef.current = null;
      }
      if (playbackAudioContextRef.current) {
        playbackAudioContextRef.current.close();
        playbackAudioContextRef.current = null;
      }
    } else {
      console.log("togglePlayback: Starting playback.");

      if (!playbackAudioContextRef.current || playbackAudioContextRef.current.state === "closed") {
        playbackAudioContextRef.current = new window.AudioContext();
        playbackAnalyserRef.current = playbackAudioContextRef.current.createAnalyser();
        playbackAnalyserRef.current.fftSize = 2048;
      }

      const currentAudioContext = playbackAudioContextRef.current;
      const currentAnalyser = playbackAnalyserRef.current;

      if (!currentAudioContext || !currentAnalyser) {
        console.error(
          "togglePlayback: AudioContext or AnalyserNode not initialized after attempt."
        );
        return;
      }

      playbackSourceNodeRef.current = currentAudioContext.createBufferSource();
      playbackSourceNodeRef.current.buffer = reviewAudioBufferRef.current;
      playbackSourceNodeRef.current.connect(currentAnalyser);
      currentAnalyser.connect(currentAudioContext.destination);

      const startTime = audioRef.current?.currentTime || 0;
      playbackSourceNodeRef.current.start(0, startTime);
      audioRef.current?.play();
      setIsPlayingBack(true);

      setTotalAudioDuration(reviewAudioBufferRef.current.duration);
      console.log(
        "togglePlayback: Playback started. Total Duration:",
        reviewAudioBufferRef.current.duration,
        "Current Time:",
        startTime
      );
    }
  };

  // Effect to update the timer every second
  useEffect(() => {
    if (recordingPhase === "recording") {
      timerTimeout = setTimeout(() => {
        setTimer(timer + 1);
      }, 1000);
    }
    return () => clearTimeout(timerTimeout);
  }, [recordingPhase, timer]);

  // Visualizer
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;

    if (!canvasCtx) return;

    const visualizeLiveVolume = () => {
      const analyser = mediaRecorderRef.current?.analyser;
      // *** MODIFICATION: Live visualizer now relies directly on the analyser, not on mediaRecorderRef.current.mediaRecorder ***
      if (!analyser || recordingPhase !== "recording") {
        if (liveAnimationRef.current) {
          cancelAnimationFrame(liveAnimationRef.current);
          liveAnimationRef.current = null;
        }
        return;
      }

      const streamSettings = mediaRecorderRef.current?.stream?.getAudioTracks()[0]?.getSettings();
      if (!streamSettings || !streamSettings.sampleRate) {
        if (liveAnimationRef.current) {
          cancelAnimationFrame(liveAnimationRef.current);
          liveAnimationRef.current = null;
        }
        return;
      }

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        if (recordingPhase !== "recording" || !analyser) {
          if (liveAnimationRef.current) {
            cancelAnimationFrame(liveAnimationRef.current);
            liveAnimationRef.current = null;
          }
          return;
        }
        liveAnimationRef.current = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);
        drawLiveWaveform(canvasCtx, WIDTH, HEIGHT, dataArray);
      };

      draw();
    };

    const visualizePlayback = () => {
      const audioBuffer = reviewAudioBufferRef.current;
      if (!audioBuffer || recordingPhase !== "review") {
        if (reviewAnimationRef.current) {
          cancelAnimationFrame(reviewAnimationRef.current);
          reviewAnimationRef.current = null;
        }
        return;
      }

      const draw = () => {
        if (recordingPhase !== "review" || !isPlayingBack || !audioBuffer) {
          if (reviewAnimationRef.current) {
            cancelAnimationFrame(reviewAnimationRef.current);
            reviewAnimationRef.current = null;
          }
          drawReviewWaveform(canvasCtx, WIDTH, HEIGHT, audioBuffer, 0, 0);
          return;
        }

        if (audioRef.current) {
          setCurrentPlaybackTime(audioRef.current.currentTime);
        }

        reviewAnimationRef.current = requestAnimationFrame(draw);
        drawReviewWaveform(
          canvasCtx,
          WIDTH,
          HEIGHT,
          audioBuffer,
          currentPlaybackTime,
          totalAudioDuration
        );
      };
      draw();
    };

    // Main logic based on recordingPhase
    if (recordingPhase === "recording") {
      console.log("Visualizer useEffect: Drawing live waveform.");
      visualizeLiveVolume();
    } else if (recordingPhase === "review") {
      const audioBuffer = reviewAudioBufferRef.current;
      if (audioBuffer) {
        console.log("Visualizer useEffect: Review phase, audioBuffer available.", {
          isPlayingBack,
          currentPlaybackTime,
          totalAudioDuration,
        });
        if (isPlayingBack) {
          console.log("Visualizer useEffect: Playing back, starting playback visualization.");
          visualizePlayback();
        } else {
          console.log("Visualizer useEffect: Not playing back, drawing static waveform.");
          if (reviewAnimationRef.current) {
            cancelAnimationFrame(reviewAnimationRef.current);
            reviewAnimationRef.current = null;
          }
          drawReviewWaveform(canvasCtx, WIDTH, HEIGHT, audioBuffer);
        }
      } else {
        console.log(
          "Visualizer useEffect: Review phase, but audioBuffer is null. Clearing canvas."
        );
        clearCanvas();
      }
    } else {
      // recordingPhase === 'idle'
      console.log("Visualizer useEffect: Idle phase. Clearing canvas.");
      clearCanvas();
    }

    return () => {
      console.log("Visualizer useEffect cleanup.");
      if (liveAnimationRef.current) cancelAnimationFrame(liveAnimationRef.current);
      if (reviewAnimationRef.current) cancelAnimationFrame(reviewAnimationRef.current);
      if (playbackAudioContextRef.current) {
        playbackAudioContextRef.current.close();
        playbackAudioContextRef.current = null;
      }
    };
  }, [
    recordingPhase,
    isPlayingBack,
    currentPlaybackTime,
    totalAudioDuration,
    finalAudioBlob,
    theme,
  ]);

  return (
    <div
      className={cn(
        "flex h-16 rounded-md relative w-full items-center justify-center gap-2 max-w-5xl",
        {
          "border p-1": recordingPhase !== "idle" && recordingPhase !== "loading_ffmpeg",
          "border-none p-0": recordingPhase === "idle" || recordingPhase === "loading_ffmpeg",
        },
        className
      )}
    >
      {recordingPhase === "loading_ffmpeg" && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm z-10 rounded-md">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          <span className="text-sm text-muted-foreground">
            {ffmpegLoadingMessage || "Loading FFmpeg..."}
          </span>
        </div>
      )}

      {recordingPhase === "converting" && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm z-10 rounded-md">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          <span className="text-sm text-muted-foreground">
            {ffmpegLoadingMessage || "Converting audio..."}
          </span>
        </div>
      )}

      {recordingPhase === "recording" && (
        <>
          <Timer
            hourLeft={hourLeft}
            hourRight={hourRight}
            minuteLeft={minuteLeft}
            minuteRight={minuteRight}
            secondLeft={secondLeft}
            secondRight={secondRight}
            timerClassName={timerClassName}
          />
          <div
            className={cn(
              "items-center -top-12 right-0 absolute justify-center gap-0.5 border p-1.5 rounded-md font-mono font-medium text-foreground flex",
              timerClassName
            )}
          >
            Listening...
          </div>
        </>
      )}

      {recordingPhase === "review" && (
        <PlaybackDisplay
          currentTime={currentPlaybackTime}
          totalDuration={totalAudioDuration}
          timerClassName={timerClassName}
        />
      )}

      {(recordingPhase === "recording" || recordingPhase === "review") && (
        <canvas ref={canvasRef} className={`h-full w-full bg-background flex`} />
      )}
      <div className="flex gap-2">
        {(recordingPhase === "idle" || recordingPhase === "loading_ffmpeg") && (
          <>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={startRecording}
                  size={"icon"}
                  disabled={!ffmpegLoaded || recordingPhase === "loading_ffmpeg"}
                >
                  {recordingPhase === "loading_ffmpeg" ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Mic size={15} />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent className="m-2">
                <span>
                  {" "}
                  {recordingPhase === "loading_ffmpeg"
                    ? "Loading FFmpeg..."
                    : "Start Listening"}{" "}
                </span>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  size={"icon"}
                  disabled={!ffmpegLoaded || recordingPhase === "loading_ffmpeg"}
                >
                  <Upload size={15} />
                </Button>
              </TooltipTrigger>
              <TooltipContent className="m-2">
                <span>Upload Audio File</span>
              </TooltipContent>
            </Tooltip>
          </>
        )}

        {recordingPhase === "recording" && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={stopListening}
                size={"icon"}
                variant={"destructive"}
                className="mr-2"
              >
                <CircleStop size={15} />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="m-2">
              <span> Stop Listening</span>
            </TooltipContent>
          </Tooltip>
        )}

        {recordingPhase === "review" && (
          <>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button onClick={togglePlayback} size={"icon"}>
                  {isPlayingBack ? <Pause size={15} /> : <Play size={15} />}
                </Button>
              </TooltipTrigger>
              <TooltipContent className="m-2">
                <span> {isPlayingBack ? "Pause Playback" : "Start Playback"} </span>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button onClick={discardRecording} size={"icon"} variant={"destructive"}>
                  <Trash size={15} />
                </Button>
              </TooltipTrigger>
              <TooltipContent className="m-2">
                <span> Discard Recording</span>
              </TooltipContent>
            </Tooltip>
          </>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
};

// New component for displaying playback duration
const PlaybackDisplay = React.memo(
  ({
    currentTime,
    totalDuration,
    timerClassName,
  }: {
    currentTime: number;
    totalDuration: number;
    timerClassName?: string;
  }) => {
    const formatTime = (timeInSeconds: number) => {
      if (!Number.isFinite(timeInSeconds) || timeInSeconds < 0) {
        return "00:00:00";
      }

      const hours = Math.floor(timeInSeconds / 3600);
      const minutes = Math.floor((timeInSeconds % 3600) / 60);
      const seconds = Math.floor(timeInSeconds % 60);

      const pad = (num: number) => String(num).padStart(2, "0");

      return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    };

    return (
      <main>
        <div
          className={cn(
            "items-center -top-12 left-0 absolute justify-center gap-0.5 border p-1.5 rounded-md font-mono font-medium text-foreground flex",
            timerClassName
          )}
        >
          <span>{formatTime(currentTime)}</span>
          <span> / </span>
          <span>{formatTime(totalDuration)}</span>
        </div>
      </main>
    );
  }
);
PlaybackDisplay.displayName = "PlaybackDisplay";

const Timer = React.memo(
  ({
    hourLeft,
    hourRight,
    minuteLeft,
    minuteRight,
    secondLeft,
    secondRight,
    timerClassName,
  }: {
    hourLeft: string;
    hourRight: string;
    minuteLeft: string;
    minuteRight: string;
    secondLeft: string;
    secondRight: string;
    timerClassName?: string;
  }) => {
    return (
      <main>
        <div
          className={cn(
            "items-center -top-12 left-0 absolute justify-center gap-0.5 border p-1.5 rounded-md font-mono font-medium text-foreground flex",
            timerClassName
          )}
        >
          <span className="rounded-md bg-background p-0.5 text-foreground">{hourLeft}</span>
          <span className="rounded-md bg-background p-0.5 text-foreground">{hourRight}</span>
          <span>:</span>
          <span className="rounded-md bg-background p-0.5 text-foreground">{minuteLeft}</span>
          <span className="rounded-md bg-background p-0.5 text-foreground">{minuteRight}</span>
          <span>:</span>
          <span className="rounded-md bg-background p-0.5 text-foreground">{secondLeft}</span>
          <span className="rounded-md bg-background p-0.5 text-foreground ">{secondRight}</span>
        </div>
      </main>
    );
  }
);
Timer.displayName = "Timer";
