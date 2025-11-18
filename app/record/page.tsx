import { AudioRecorderWithVisualizer } from "@/components/audio-recorder-visualizer";

export default function Home() {
  return (
    <main className="h-screen relative">
      <div className="flex justify-center items-center h-full w-full bg max-w-md mx-auto ">
        <AudioRecorderWithVisualizer />
      </div>
    </main>
  );
}
