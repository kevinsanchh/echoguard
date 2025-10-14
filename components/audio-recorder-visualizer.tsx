"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { CircleStop, Mic, Play, Pause, Trash } from "lucide-react";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";

type Props = {
  className?: string;
  timerClassName?: string;
};

// Global timeouts/intervals (managed by refs now for better cleanup)
let timerTimeout: NodeJS.Timeout;

// Utility function to pad a number with leading zeros
const padWithLeadingZeros = (num: number, length: number): string => {
  return String(num).padStart(length, "0");
};

type RecordingPhase = "idle" | "recording" | "review";

// Configuration for backend audio clips
const CLIP_DURATION_MS = 5000; // 5 seconds per clip

export const AudioRecorderWithVisualizer = ({ className, timerClassName }: Props) => {
  const { theme } = useTheme();
  // States
  const [recordingPhase, setRecordingPhase] = useState<RecordingPhase>("idle");
  const [isRecording, setIsRecording] = useState<boolean>(false); // Derived from recordingPhase, but kept for visualizer logic
  const [timer, setTimer] = useState<number>(0);
  const [finalAudioBlob, setFinalAudioBlob] = useState<Blob | null>(null);
  const [isPlayingBack, setIsPlayingBack] = useState<boolean>(false);
  const [currentPlaybackTime, setCurrentPlaybackTime] = useState<number>(0);
  const [totalAudioDuration, setTotalAudioDuration] = useState<number>(0);

  // Refs for the primary MediaStream and its AudioContext/Analyser for LIVE visualization
  const mediaRecorderRef = useRef<{
    stream: MediaStream | null;
    analyser: AnalyserNode | null;
    // mediaRecorder: MediaRecorder | null; // Removed - no dedicated visualizer recorder needed
    audioContext: AudioContext | null; // For live recording analysis
  }>({
    stream: null,
    analyser: null,
    // mediaRecorder: null, // Removed
    audioContext: null,
  });

  // Refs for backend continuous clipping
  const backendClipRecorderRef = useRef<MediaRecorder | null>(null);
  const backendClipChunks = useRef<BlobPart[]>([]); // Chunks for the CURRENT backend clip
  const backendRecorderTimeoutId = useRef<NodeJS.Timeout | null>(null);

  // *** NEW: Ref to store history of completed backend clips for frontend playback ***

  const audioRef = useRef<HTMLAudioElement | null>(null); // For playback
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const liveAnimationRef = useRef<number | null>(null); // For live recording waveform
  const reviewAnimationRef = useRef<number | null>(null); // For review waveform

  // Refs for review/playback analysis
  const playbackAudioContextRef = useRef<AudioContext | null>(null);
  const playbackSourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const playbackAnalyserRef = useRef<AnalyserNode | null>(null);
  const reviewAudioBufferRef = useRef<AudioBuffer | null>(null); // To store decoded audio buffer for review

  const latestRecordingPhase = useRef<RecordingPhase>("idle"); // To store the latest recordingPhase

  // Sync latestRecordingPhase ref with recordingPhase state
  useEffect(() => {
    latestRecordingPhase.current = recordingPhase;
  }, [recordingPhase]);

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

  const sendAudioToBackend = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, `audio_${Date.now()}.webm`); // Filename and type to match WebM

    try {
      console.log(
        `Attempting to send audio clip (${(audioBlob.size / 1024).toFixed(
          2
        )} KB) to /api/audio-upload`
      );
      const response = await fetch("/api/audio-upload", {
        method: "POST",
        body: formData,
      });

      console.log(
        "Response received for audio upload. Status:",
        response.status,
        "URL:",
        response.url
      );

      if (response.ok) {
        console.log("Audio clip sent successfully! Backend response status:", response.status);
        try {
          const responseData = await response.json();
          console.log("Backend JSON response data:", responseData);
          // TODO: Use backend response (e.g., prediction) here for future flags
        } catch (jsonError) {
          console.warn(
            "Could not parse JSON response from backend (might be empty or HTML from redirect).",
            jsonError
          );
          const textResponse = await response.text();
          console.log(
            "Backend response as text (first 200 chars):",
            textResponse.substring(0, 200) + "..."
          );
        }
      } else {
        let errorData = {};
        try {
          errorData = await response.json();
        } catch (e) {
          errorData = { message: await response.text() };
        }
        console.error(
          "Failed to send audio clip:",
          response.status,
          response.statusText,
          errorData
        );
      }
    } catch (error) {
      console.error("Error sending audio clip (network or CORS issue):", error);
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
  const _startBackendRecordingCycle = (stream: MediaStream) => {
    _stopBackendRecordingCycle(); // Clean up any previous cycle's timeout or recorder instance

    if (!stream.active) {
      console.warn("Attempted to start backend recording cycle with an inactive stream. Skipping.");
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
        const fullClipBlob = new Blob(backendClipChunks.current, { type: mimeType });
        await sendAudioToBackend(fullClipBlob);
        // *** MODIFICATION: Removed line pushing to frontendPlaybackClipHistoryRef.current ***
        // console.log(`Frontend playback history updated. Total clips: ${frontendPlaybackClipHistoryRef.current.length}`); // Log removed
      }
      if (latestRecordingPhase.current === "recording") {
        console.log("Restarting backend recording cycle...");
        _startBackendRecordingCycle(stream);
      } else {
        console.log(
          "Not restarting backend recording cycle, latestRecordingPhase is not 'recording'. Current phase:",
          latestRecordingPhase.current
        );
      }
    };

    backendClipRecorderRef.current.onerror = (event) => {
      console.error("Backend MediaRecorder error:", event);
      if (latestRecordingPhase.current === "recording") {
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

  function startRecording() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          setRecordingPhase("recording");
          setIsRecording(true);
          setTimer(0);
          // *** MODIFICATION: Clear previous final blob and history ***
          setFinalAudioBlob(null); // Clear any previous stitched audio
          // frontendPlaybackClipHistoryRef.current = []; // Removed - no longer managing history on frontend
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
          };

          let mimeType = MediaRecorder.isTypeSupported("audio/webm")
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
          console.log(`MediaRecorder for live visualizer will use MIME type: ${mimeType}`);

          _startBackendRecordingCycle(stream);
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
    const { stream, analyser, audioContext } = mediaRecorderRef.current;

    _stopBackendRecordingCycle();

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (analyser) {
      analyser.disconnect();
    }
    if (audioContext) {
      audioContext.close();
    }
    if (liveAnimationRef.current) {
      cancelAnimationFrame(liveAnimationRef.current);
    }

    setIsRecording(false);
    clearTimeout(timerTimeout);

    console.log("stopListening: Requesting combined audio from backend for playback.");
    try {
      const stitchResponse = await fetch("/api/stitch-and-return-audio", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (stitchResponse.ok) {
        const data = await stitchResponse.json();
        if (data.audioDataUri) {
          console.log("stopListening: Received combined audio data URI from backend.");
          const blob = await fetch(data.audioDataUri).then((res) => res.blob());
          setFinalAudioBlob(blob);

          const tempAudioContext = new window.AudioContext();
          blob
            .arrayBuffer()
            .then((buffer) => {
              return tempAudioContext.decodeAudioData(buffer);
            })
            .then((audioBuffer) => {
              reviewAudioBufferRef.current = audioBuffer;
              setTotalAudioDuration(audioBuffer.duration);
              tempAudioContext.close();
              setRecordingPhase("review");
              console.log(
                "stopListening: AudioBuffer decoded from stitched backend audio, transitioning to review."
              );
              // *** NEW LOGS: Inspect AudioBuffer contents ***
              console.log("AudioBuffer properties:", {
                duration: audioBuffer.duration,
                sampleRate: audioBuffer.sampleRate,
                numberOfChannels: audioBuffer.numberOfChannels,
                firstChannelDataLength: audioBuffer.getChannelData(0).length,
                first5Samples: Array.from(audioBuffer.getChannelData(0).slice(0, 5)),
              });
              if (audioBuffer.getChannelData(0).some((sample) => sample !== 0)) {
                console.log("AudioBuffer contains non-zero data (likely audible).");
              } else {
                console.warn("AudioBuffer contains only zero data (likely silent or corrupted).");
              }
              // *** END NEW LOGS ***
            })
            .catch((e) => {
              console.error(
                "stopListening: Error decoding stitched audio for duration/waveform:",
                e
              );
              tempAudioContext.close();
              setTotalAudioDuration(0);
              setRecordingPhase("idle");
              setFinalAudioBlob(null);
              reviewAudioBufferRef.current = null;
            });
        } else {
          console.error(
            "stopListening: Backend did not return audioDataUri in a successful response."
          );
          setRecordingPhase("idle");
          setTotalAudioDuration(0);
          setFinalAudioBlob(null);
          reviewAudioBufferRef.current = null;
        }
      } else {
        const errorText = await stitchResponse.text();
        let errorData = { message: errorText };

        try {
          const jsonError = JSON.parse(errorText);
          errorData = jsonError;
        } catch (e) {
          // Not JSON
        }

        console.error(
          "stopListening: Failed to get stitched audio from backend:",
          stitchResponse.status,
          stitchResponse.statusText,
          errorData
        );
        setRecordingPhase("idle");
        setTotalAudioDuration(0);
        setFinalAudioBlob(null);
        reviewAudioBufferRef.current = null;
      }
    } catch (error) {
      console.error("stopListening: Network error requesting stitched audio:", error);
      setRecordingPhase("idle");
      setTotalAudioDuration(0);
      setFinalAudioBlob(null);
      reviewAudioBufferRef.current = null;
    }

    mediaRecorderRef.current = {
      stream: null,
      analyser: null,
      audioContext: null,
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
    // *** MODIFICATION: Removed frontendPlaybackClipHistoryRef.current = []; ***
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
      _stopBackendRecordingCycle();
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
          "border p-1": recordingPhase !== "idle", // Apply border if not idle
          "border-none p-0": recordingPhase === "idle", // No border if idle
        },
        className
      )}
    >
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
        {recordingPhase === "idle" && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button onClick={startRecording} size={"icon"}>
                <Mic size={15} />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="m-2">
              <span> Start Listening</span>
            </TooltipContent>
          </Tooltip>
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
