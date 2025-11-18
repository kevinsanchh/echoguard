"use client";
import { AudioRecorderWithVisualizer } from "@/components/audio-recorder-visualizer";
import DashboardDialog from "@/components/dashboard-dialog";
import { LogoIcon } from "@/components/icons/logo-icon";
import { Mic, Settings2 } from "lucide-react";
import { useState } from "react";

export default function Home() {
  const [open, setOpen] = useState(false);
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const toggleDashboard = () => setIsDashboardOpen(!isDashboardOpen);

  return (
    <main className="h-screen relative">
      <div className="flex gap-2 flex-col justify-center items-center h-full w-full bg max-w-xs  mx-auto ">
        {/* <AudioRecorderWithVisualizer /> */}
        <div className="border text-[#3d98f6] border-neutral-200 rounded-md p-2 flex items-center justify-center gap-2 w-full">
          <LogoIcon className="text-[#3d98f6] drop-shadow-sm" />
          <h1 className="font-medium drop-shadow-sm">EchoGuard</h1>
        </div>
        <div className="border border-neutral-200 rounded-md p-2 grid grid-cols-2 gap-2 w-full">
          <a
            href="/record"
            className="bg-green-50 cursor-pointer hover:scale-[1.03] duration-75 p-4 border border-dashed border-green-600 flex flex-col justify-center items-center gap-2 rounded-md "
          >
            <div className="bg-green-600 rounded-md p-1 ">
              <Mic size={18} className="text-green-50" />
            </div>
            <h1 className="text-sm text-green-600 font-[450]">Start Monitoring</h1>
          </a>
          <div
            onClick={toggleDashboard}
            className="bg-neutral-100 cursor-pointer hover:scale-[1.03] duration-75 p-4 border border-dashed border-neutral-400 flex flex-col justify-center items-center gap-2 rounded-md "
          >
            <div className="bg-neutral-600 rounded-md p-1 ">
              <Settings2 size={18} className="text-green-50" />
            </div>
            <h1 className="text-sm text-neutral-600 font-[450]">Dashboard</h1>
          </div>
          <DashboardDialog toggleModal={toggleDashboard} externalIsOpen={isDashboardOpen} />
        </div>
      </div>
    </main>
  );
}
