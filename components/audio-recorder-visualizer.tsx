"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { CircleStop, Download, Mic, Trash } from "lucide-react";
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

export const AudioRecorderWithVisualizer = ({ className, timerClassName }: Props) => {
  const { theme } = useTheme();
  // States
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [timer, setTimer] = useState<number>(0);
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
  // Refs
  const mediaRecorderRef = useRef<{
    stream: MediaStream | null;
    analyser: AnalyserNode | null;
    mediaRecorder: MediaRecorder | null;
    audioContext: AudioContext | null;
  }>({
    stream: null,
    analyser: null,
    mediaRecorder: null,
    audioContext: null,
  });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<any>(null);

  const sendAudioToBackend = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, `audio_${Date.now()}.wav`);

    try {
      console.log("Attempting to send audio clip to /api/audio-upload"); // Debug log
      const response = await fetch("/api/audio-upload", {
        method: "POST",
        body: formData,
      });

      console.log(
        "Response received for audio upload. Status:",
        response.status,
        "URL:",
        response.url
      ); // More detailed debug log

      if (response.ok) {
        console.log("Audio clip sent successfully! Backend response status:", response.status);
        // Try to parse JSON. If it's a redirect returning HTML, this will fail.
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
          errorData = { message: await response.text() }; // Fallback to text if not JSON
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

  function startRecording() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          setIsRecording(true);
          setTimer(0); // Reset timer when starting continuous listening

          const AudioContext = window.AudioContext;
          const audioCtx = new AudioContext();
          const analyser = audioCtx.createAnalyser();
          const source = audioCtx.createMediaStreamSource(stream);
          source.connect(analyser);

          mediaRecorderRef.current = {
            stream,
            analyser,
            mediaRecorder: null, // Will be set below
            audioContext: audioCtx,
          };

          // Prefer webm for better compression and browser support, fallback to mpeg or wav
          const mimeType = MediaRecorder.isTypeSupported("audio/webm")
            ? "audio/webm"
            : MediaRecorder.isTypeSupported("audio/mpeg")
            ? "audio/mpeg"
            : "audio/wav";

          const options = { mimeType };
          const mediaRecorder = new MediaRecorder(stream, options);

          // This event fires every 30 seconds due to mediaRecorder.start(30000)
          mediaRecorder.ondataavailable = async (e) => {
            if (e.data.size > 0) {
              const audioBlob = new Blob([e.data], { type: mimeType });
              await sendAudioToBackend(audioBlob);
            }
          };

          mediaRecorder.onstop = () => {
            console.log("MediaRecorder stopped.");
            // Any final cleanup related to stopping can go here if needed.
          };

          mediaRecorder.start(5000); // Start recording in 30-second slices

          mediaRecorderRef.current.mediaRecorder = mediaRecorder; // Store the MediaRecorder instance in ref
        })
        .catch((error) => {
          alert("Error accessing microphone: " + error.message);
          console.error("Error accessing microphone:", error);
          setIsRecording(false); // Ensure recording state is reset on error
        });
    } else {
      alert("getUserMedia not supported on your browser!");
    }
  }
  function stopListening() {
    const { mediaRecorder, stream, analyser, audioContext } = mediaRecorderRef.current;

    if (mediaRecorder) {
      mediaRecorder.onstop = () => {
        // recordingChunks = []; // This line is no longer needed and can be removed
      };
      mediaRecorder.stop();
    } else {
      alert("recorder instance is null!");
    }

    // Stop the web audio context and the analyser node
    if (analyser) {
      analyser.disconnect();
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (audioContext) {
      audioContext.close();
    }
    setIsRecording(false);
    setTimer(0);
    clearTimeout(timerTimeout);

    // Clear the animation frame and canvas
    cancelAnimationFrame(animationRef.current || 0);
    const canvas = canvasRef.current;
    if (canvas) {
      const canvasCtx = canvas.getContext("2d");
      if (canvasCtx) {
        const WIDTH = canvas.width;
        const HEIGHT = canvas.height;
        canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);
      }
    }
  }

  // Effect to update the timer every second
  useEffect(() => {
    if (isRecording) {
      timerTimeout = setTimeout(() => {
        setTimer(timer + 1);
      }, 1000);
    }
    return () => clearTimeout(timerTimeout);
  }, [isRecording, timer]);

  // Visualizer
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;

    const drawWaveform = (dataArray: Uint8Array) => {
      if (!canvasCtx) return;
      canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);
      canvasCtx.fillStyle = "#939393";

      const barWidth = 1;
      const spacing = 1;
      const maxBarHeight = HEIGHT / 2.5;
      const numBars = Math.floor(WIDTH / (barWidth + spacing));

      for (let i = 0; i < numBars; i++) {
        const barHeight = Math.pow(dataArray[i] / 128.0, 8) * maxBarHeight;
        const x = (barWidth + spacing) * i;
        const y = HEIGHT / 2 - barHeight / 2;
        canvasCtx.fillRect(x, y, barWidth, barHeight);
      }
    };

    const visualizeVolume = () => {
      if (!mediaRecorderRef.current?.stream?.getAudioTracks()[0]?.getSettings().sampleRate) return;
      const bufferLength =
        (mediaRecorderRef.current?.stream?.getAudioTracks()[0]?.getSettings()
          .sampleRate as number) / 100;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        if (!isRecording) {
          cancelAnimationFrame(animationRef.current || 0);
          return;
        }
        animationRef.current = requestAnimationFrame(draw);
        mediaRecorderRef.current?.analyser?.getByteTimeDomainData(dataArray);
        drawWaveform(dataArray);
      };

      draw();
    };

    if (isRecording) {
      visualizeVolume();
    } else {
      if (canvasCtx) {
        canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);
      }
      cancelAnimationFrame(animationRef.current || 0);
    }

    return () => {
      cancelAnimationFrame(animationRef.current || 0);
    };
  }, [isRecording, theme]);

  return (
    <div
      className={cn(
        "flex h-16 rounded-md relative w-full items-center justify-center gap-2 max-w-5xl",
        {
          "border p-1": isRecording,
          "border-none p-0": !isRecording,
        },
        className
      )}
    >
      {isRecording ? (
        <Timer
          hourLeft={hourLeft}
          hourRight={hourRight}
          minuteLeft={minuteLeft}
          minuteRight={minuteRight}
          secondLeft={secondLeft}
          secondRight={secondRight}
          timerClassName={timerClassName}
        />
      ) : null}

      {/* New: "Recording..." box */}
      {isRecording ? (
        <div
          className={cn(
            "items-center -top-12 right-0 absolute justify-center gap-0.5 border p-1.5 rounded-md font-mono font-medium text-foreground flex",
            timerClassName // Apply the same timerClassName for consistent styling
          )}
        >
          Listening...
        </div>
      ) : null}

      <canvas
        ref={canvasRef}
        className={`h-full w-full bg-background ${!isRecording ? "hidden" : "flex"}`}
      />
      <div className="flex gap-2">
        {/* ========== Start and Stop Listening button ========== */}
        <Tooltip>
          <TooltipTrigger asChild>
            {!isRecording ? (
              <Button onClick={startRecording} size={"icon"}>
                <Mic size={15} />
              </Button>
            ) : (
              <Button
                onClick={stopListening}
                size={"icon"}
                variant={"destructive"}
                className="mr-2"
              >
                <CircleStop size={15} />
              </Button>
            )}
          </TooltipTrigger>
          <TooltipContent className="m-2">
            <span> {!isRecording ? "Start Listening" : "Stop Listening"} </span>
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
};

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
