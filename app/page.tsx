import { AudioRecorderWithVisualizer } from "@/components/audio-recorder-visualizer";

export default function Home() {
  return (
    <main className="max-w-md  mx-auto  h-screen">
      <div className="flex justify-center items-center h-full w-full bg ">
        <AudioRecorderWithVisualizer />
      </div>
    </main>
  );
}
