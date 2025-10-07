"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { CircleStop, Mic, Play, Pause, Trash } from "lucide-react"; // Added Play, Pause, Trash icons
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";

type Props = {
  className?: string;
  timerClassName?: string;
};

let timerTimeout: NodeJS.Timeout;

// Utility function to pad a number with leading zeros
const padWithLeadingZeros = (num: number, length: number): string => {
  return String(num).padStart(length, "0");
};

type RecordingPhase = "idle" | "recording" | "review";

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

  // Refs for MediaRecorder and AudioContext instances
  const mediaRecorderRef = useRef<{
    stream: MediaStream | null;
    analyser: AnalyserNode | null;
    mediaRecorder: MediaRecorder | null;
    audioContext: AudioContext | null; // For live recording analysis
  }>({
    stream: null,
    analyser: null,
    mediaRecorder: null,
    audioContext: null,
  });

  // Refs for collected audio chunks and playback
  const allRecordedChunks = useRef<BlobPart[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null); // For playback
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const liveAnimationRef = useRef<number | null>(null); // For live recording waveform
  const reviewAnimationRef = useRef<number | null>(null); // For review waveform
  const reviewAudioBufferRef = useRef<AudioBuffer | null>(null); // To store decoded audio buffer for review

  // Refs for review/playback analysis
  const playbackAudioContextRef = useRef<AudioContext | null>(null);
  const playbackSourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const playbackAnalyserRef = useRef<AnalyserNode | null>(null);

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
    formData.append("audio", audioBlob, `audio_${Date.now()}.wav`);

    try {
      console.log("Attempting to send audio clip to /api/audio-upload");
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

  function startRecording() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          setRecordingPhase("recording");
          setIsRecording(true); // Update isRecording for visualizer useEffect
          setTimer(0);
          allRecordedChunks.current = []; // Clear previous chunks

          const AudioContext = window.AudioContext;
          const audioCtx = new AudioContext();
          const analyser = audioCtx.createAnalyser();
          const source = audioCtx.createMediaStreamSource(stream);
          source.connect(analyser);

          mediaRecorderRef.current = {
            stream,
            analyser,
            mediaRecorder: null,
            audioContext: audioCtx,
          };

          const mimeType = MediaRecorder.isTypeSupported("audio/webm")
            ? "audio/webm"
            : MediaRecorder.isTypeSupported("audio/mpeg")
            ? "audio/mpeg"
            : "audio/wav";

          const options = { mimeType };
          const mediaRecorder = new MediaRecorder(stream, options);

          mediaRecorder.ondataavailable = async (e) => {
            if (e.data.size > 0) {
              allRecordedChunks.current.push(e.data); // Store for final playback
              const audioBlob = new Blob([e.data], { type: mimeType });
              await sendAudioToBackend(audioBlob);
            }
          };

          mediaRecorder.onstop = () => {
            console.log("MediaRecorder stopped.");
            // This onstop is triggered when `mediaRecorder.stop()` is explicitly called
            // No direct action here, as stopListening handles the state transition.
          };

          mediaRecorder.start(5000); // Start recording in 5-second slices (adjust to 30000 for 30s)

          mediaRecorderRef.current.mediaRecorder = mediaRecorder;
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

  function stopListening() {
    const { mediaRecorder, stream, analyser, audioContext } = mediaRecorderRef.current;

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
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

    if (allRecordedChunks.current.length > 0) {
      const mimeType =
        allRecordedChunks.current[0] instanceof Blob
          ? allRecordedChunks.current[0].type
          : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "audio/wav";

      const fullBlob = new Blob(allRecordedChunks.current, { type: mimeType });
      setFinalAudioBlob(fullBlob);

      const tempAudioContext = new window.AudioContext();
      fullBlob
        .arrayBuffer()
        .then((buffer) => {
          tempAudioContext
            .decodeAudioData(buffer)
            .then((audioBuffer) => {
              reviewAudioBufferRef.current = audioBuffer; // Store for later
              setTotalAudioDuration(audioBuffer.duration); // Set total duration immediately
              tempAudioContext.close();
              setRecordingPhase("review"); // Transition to review phase after duration is known
              console.log(
                "stopListening: AudioBuffer decoded, duration set, transitioning to review. Buffer:",
                reviewAudioBufferRef.current
              ); // Debug log
            })
            .catch((e) => {
              console.error("stopListening: Error decoding audio for duration display:", e);
              tempAudioContext.close();
              setTotalAudioDuration(0);
              setRecordingPhase("review");
            });
        })
        .catch((e) => {
          console.error("stopListening: Error reading audio blob for duration display:", e);
          setTotalAudioDuration(0);
          setRecordingPhase("review");
        });
    } else {
      console.warn("stopListening: No audio chunks recorded to review.");
      setRecordingPhase("idle");
      setTotalAudioDuration(0);
    }

    mediaRecorderRef.current = {
      stream: null,
      analyser: null,
      mediaRecorder: null,
      audioContext: null,
    };
  }

  function discardRecording() {
    // Stop any ongoing playback
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
    }
    setIsPlayingBack(false);
    setCurrentPlaybackTime(0); // Reset current playback time
    setTotalAudioDuration(0); // Reset total audio duration

    // Clear any previous recorded data
    setFinalAudioBlob(null);
    allRecordedChunks.current = [];
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
        setCurrentPlaybackTime(0); // Reset to 0 when playback ends
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
      // REMOVED: audioRef.current.ontimeupdate = ...
    }

    // Always update audioRef.current.src here if it's not already set to the current blob
    if (!audioRef.current.src || audioRef.current.src !== URL.createObjectURL(finalAudioBlob)) {
      if (audioRef.current.src) URL.revokeObjectURL(audioRef.current.src); // Clean up old URL if present
      audioRef.current.src = URL.createObjectURL(finalAudioBlob);
      audioRef.current.load(); // Load the new source
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
      playbackSourceNodeRef.current.buffer = reviewAudioBufferRef.current; // Use the stored buffer
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

  // Drawing utilities for the canvas
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
    const maxBarHeight = HEIGHT / 2.5;
    const numBars = Math.floor(WIDTH / (barWidth + spacing));

    for (let i = 0; i < numBars; i++) {
      const dataIndex = Math.floor((i * dataArray.length) / numBars); // Downsample for drawing efficiency
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
    currentPlaybackTime: number = 0, // New optional parameter
    totalAudioDuration: number = 0 // New optional parameter
  ) => {
    canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);
    canvasCtx.strokeStyle = "#939393";
    canvasCtx.lineWidth = 2;

    const channelData = audioBuffer.getChannelData(0); // Use first channel for waveform

    // Downsample for drawing efficiency, drawing more samples than pixels for detail
    const samplesToDraw = WIDTH * 2;
    const step = Math.floor(channelData.length / samplesToDraw);

    canvasCtx.beginPath();
    for (let i = 0; i < samplesToDraw; i++) {
      const x = (i / samplesToDraw) * WIDTH;
      // Normalize audio data from [-1, 1] to canvas y-coordinates [0, HEIGHT]
      const y = (0.5 + channelData[i * step]) * HEIGHT;
      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
    }
    canvasCtx.stroke();

    // --- Draw Playback Head (vertical line) ---
    if (currentPlaybackTime > 0 && totalAudioDuration > 0) {
      const playbackPositionX = (currentPlaybackTime / totalAudioDuration) * WIDTH;
      canvasCtx.beginPath();
      canvasCtx.strokeStyle = "#0084D1"; // Or any distinct color for the playback head
      canvasCtx.lineWidth = 2;
      canvasCtx.moveTo(playbackPositionX, 0);
      canvasCtx.lineTo(playbackPositionX, HEIGHT);
      canvasCtx.stroke();
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

  // Visualizer Logic
  // Visualizer
  // Visualizer
  // Visualizer
  // Visualizer
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;

    if (!canvasCtx) return; // Ensure context is available

    const visualizeLiveVolume = () => {
      const analyser = mediaRecorderRef.current?.analyser;
      if (!analyser) {
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

      const bufferLength = analyser.frequencyBinCount; // Using frequencyBinCount for time domain
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

        // *** NEW: Update currentPlaybackTime directly from audioRef in animation loop ***
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
      console.log("Visualizer useEffect: Drawing live waveform."); // Debug log
      visualizeLiveVolume();
    } else if (recordingPhase === "review") {
      const audioBuffer = reviewAudioBufferRef.current;
      if (audioBuffer) {
        console.log("Visualizer useEffect: Review phase, audioBuffer available.", {
          isPlayingBack,
          currentPlaybackTime,
          totalAudioDuration,
        }); // Debug log
        if (isPlayingBack) {
          console.log("Visualizer useEffect: Playing back, starting playback visualization."); // Debug log
          visualizePlayback(); // This loop will continuously draw the waveform with playback head
        } else {
          console.log("Visualizer useEffect: Not playing back, drawing static waveform."); // Debug log
          if (reviewAnimationRef.current) {
            cancelAnimationFrame(reviewAnimationRef.current);
            reviewAnimationRef.current = null;
          }
          drawReviewWaveform(canvasCtx, WIDTH, HEIGHT, audioBuffer); // Draw static waveform
        }
      } else {
        console.log(
          "Visualizer useEffect: Review phase, but audioBuffer is null. Clearing canvas."
        ); // Debug log
        clearCanvas();
      }
    } else {
      // recordingPhase === 'idle'
      console.log("Visualizer useEffect: Idle phase. Clearing canvas."); // Debug log
      clearCanvas();
    }

    return () => {
      console.log("Visualizer useEffect cleanup."); // Debug log
      if (liveAnimationRef.current) cancelAnimationFrame(liveAnimationRef.current);
      if (reviewAnimationRef.current) cancelAnimationFrame(reviewAnimationRef.current);
      if (playbackAudioContextRef.current) {
        playbackAudioContextRef.current.close();
        playbackAudioContextRef.current = null;
      }
      // No longer clear reviewAudioBufferRef.current here, as it should persist across review phase renders
    };
  }, [
    recordingPhase,
    isPlayingBack,
    currentPlaybackTime,
    totalAudioDuration,
    finalAudioBlob,
    theme,
  ]); // Keep finalAudioBlob in deps as it triggers initial decode for review; // Added canvasCtx, WIDTH, HEIGHT to dependencies;

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

      {recordingPhase === "review" && ( // New condition for playback display
        <PlaybackDisplay
          currentTime={currentPlaybackTime}
          totalDuration={totalAudioDuration}
          timerClassName={timerClassName}
        />
      )}

      {(recordingPhase === "recording" || recordingPhase === "review") && (
        <canvas
          ref={canvasRef}
          className={`h-full w-full bg-background flex`} // Removed the inner conditional for 'hidden'
        />
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
      // Handle non-finite or negative numbers gracefully
      if (!Number.isFinite(timeInSeconds) || timeInSeconds < 0) {
        return "00:00:00"; // Display a default zero time for invalid values
      }

      const hours = Math.floor(timeInSeconds / 3600);
      const minutes = Math.floor((timeInSeconds % 3600) / 60);
      const seconds = Math.floor(timeInSeconds % 60);

      const pad = (num: number) => String(num).padStart(2, "0");

      return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    };

    return (
      <main>
        {" "}
        {/* Wrap in main for consistency with Timer */}
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
