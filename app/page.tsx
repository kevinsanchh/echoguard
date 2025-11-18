import { AudioRecorderWithVisualizer } from "@/components/audio-recorder-visualizer";
import DashboardDialog from "@/components/dashboard-dialog";
import { Mic, Settings2 } from "lucide-react";

export default function Home() {
  return (
    <main className="h-screen relative">
      <div className="flex justify-center items-center h-full w-full bg max-w-md mx-auto ">
        {/* <AudioRecorderWithVisualizer /> */}
        <div className="border border-neutral-200 rounded-md p-2 grid grid-cols-2 gap-2 ">
          <a
            href="/record"
            className="bg-green-50 cursor-pointer hover:scale-[1.03] duration-75 p-4 border border-dashed border-green-600 flex flex-col justify-center items-center gap-2 rounded-md "
          >
            <div className="bg-green-600 rounded-md p-1 ">
              <Mic size={18} className="text-green-50" />
            </div>
            <h1 className="text-sm text-green-600 font-[450]">Start Monitoring</h1>
          </a>
          <div className="bg-neutral-100 cursor-pointer hover:scale-[1.03] duration-75 p-4 border border-dashed border-neutral-400 flex flex-col justify-center items-center gap-2 rounded-md ">
            <div className="bg-neutral-600 rounded-md p-1 ">
              <Settings2 size={18} className="text-green-50" />
            </div>
            <h1 className="text-sm text-neutral-600 font-[450]">Dashboard</h1>
          </div>
          <DashboardDialog />
        </div>
      </div>
    </main>
  );
}
