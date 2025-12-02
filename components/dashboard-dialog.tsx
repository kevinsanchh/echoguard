"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, Variants, Transition } from "framer-motion";
import { Button } from "./ui/button";
import { AudioLines, CircleX, AlertTriangle } from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { format } from "date-fns";

const DashboardDialog = ({
  toggleModal: externalToggleModal,
  externalIsOpen,
}: {
  toggleModal?: () => void;
  externalIsOpen?: boolean;
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const isVisible = externalIsOpen !== undefined ? externalIsOpen : isOpen;

  const modalRef = useRef<HTMLDivElement>(null);

  const defaultVariants: Variants = {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
  };

  const defaultTransition: Transition = {
    ease: "easeOut",
    duration: 0.2,
  };

  const backdropVariants: Variants = {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
  };

  const toggleModal = () => {
    if (externalToggleModal) return externalToggleModal();
    setIsOpen(!isOpen);
  };

  const closeModal = () => {
    if (externalToggleModal) return externalToggleModal();
    setIsOpen(false);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        if (externalToggleModal) externalToggleModal();
        else closeModal();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  return (
    <>
      {!externalToggleModal && (
        <Button onClick={toggleModal} size="icon" variant="outline" className="border-neutral-200">
          <AudioLines size={15} />
        </Button>
      )}

      <AnimatePresence mode="wait">
        {isVisible && (
          <motion.div
            key="modal-backdrop"
            id="default-modal"
            tabIndex={-1}
            aria-hidden="true"
            variants={backdropVariants}
            transition={defaultTransition}
            initial="initial"
            animate="animate"
            exit="initial"
            className="overflow-y-auto overflow-x-hidden fixed top-0 right-0 left-0 z-50 flex justify-center items-center w-full h-[calc(100%)] max-h-full bg-black/20 backdrop-blur-xs"
          >
            <motion.div
              key="terms-modal"
              ref={modalRef}
              variants={defaultVariants}
              transition={defaultTransition}
              initial="initial"
              animate="animate"
              exit="initial"
              className="relative p-4 w-full h-full max-h-[90%] max-w-2xl aspect-3/4"
            >
              <div className="h-full w-full relative bg-white rounded-lg shadow-sm dark:bg-gray-700">
                {/* Header */}
                <div className="p-4 md:p-5 flex justify-between items-center border-b border-neutral-100">
                  <h1 className="font-semibold text-xl">Dashboard</h1>
                  <Button
                    onClick={closeModal}
                    size="icon"
                    variant="outline"
                    className="border-neutral-200 rounded-full"
                  >
                    <CircleX size={15} className="text-neutral-400" />
                  </Button>
                </div>

                {/* Body */}
                <div className="p-4 md:p-5 space-y-4 overflow-y-scroll">
                  <h3 className="font-semibold text-lg mb-2">Saved Recordings</h3>

                  <Accordion
                    type="single"
                    collapsible
                    className="flex pb-1 flex-col gap-4 overflow-y-scroll max-h-[36.8rem]"
                  >
                    {(() => {
                      if (typeof window === "undefined") return null;
                      const data = JSON.parse(localStorage.getItem("echoguard_recordings") || "[]");

                      if (data.length === 0)
                        return (
                          <AccordionItem value="empty">
                            <AccordionTrigger>No recordings yet</AccordionTrigger>
                          </AccordionItem>
                        );

                      return data
                        .slice()
                        .reverse()
                        .map((record: any, idx: number) => {
                          const rawConfidence = record?.confidence_score;
                          const confidence =
                            rawConfidence === undefined ||
                            rawConfidence === null ||
                            rawConfidence === "" ||
                            isNaN(Number(rawConfidence))
                              ? null
                              : Number(rawConfidence);

                          const isLowConfidence =
                            typeof confidence === "number" && confidence <= 0.6;

                          const confidenceColor =
                            confidence === null
                              ? "text-neutral-400"
                              : confidence <= 0.4
                              ? "text-red-600"
                              : confidence <= 0.7
                              ? "text-yellow-600"
                              : "text-green-600";

                          const firstSentenceReasoning =
                            (record.confidence_reasoning || "").split(".")[0] + ".";

                          return (
                            <AccordionItem
                              key={record.id}
                              value={`item-${idx}`}
                              className="border rounded-md cursor-pointer active:scale-[0.97] duration-100 bg-white"
                            >
                              <AccordionTrigger>
                                <div className="w-full  rounded-md p-3 bg-background/80 backdrop-blur-sm">
                                  <h1 className="font-semibold mb-3 border-b pb-2">
                                    Recording{" "}
                                    {format(new Date(record.date), "MMM. dd, yyyy h:mm a")}
                                  </h1>
                                  {record.detections.length === 0 && (
                                    <div className="col-span-3 text-neutral-500 text-center text-sm italic p-2 border border-neutral-200 rounded-md bg-neutral-50">
                                      No environmental sounds detected.
                                    </div>
                                  )}

                                  <div className="grid grid-cols-3 gap-2">
                                    {record.detections.map((det: any, i: number) => (
                                      <div
                                        key={i}
                                        className="border bg-neutral-50 border-neutral-200 rounded-md p-2"
                                      >
                                        <h1 className="text-md text-neutral-600">{det.label}</h1>
                                        <h1 className="font-semibold text-right text-xl ">
                                          {String(det.confidence).split(".")[0]}%
                                        </h1>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </AccordionTrigger>

                              <AccordionContent asChild className="flex flex-col gap-4 mx-3">
                                <motion.div
                                  initial={{ opacity: 0, height: 0 }}
                                  animate={{ opacity: 1, height: "auto" }}
                                  exit={{ opacity: 0, height: 0 }}
                                  transition={{ duration: 0.15 }}
                                >
                                  <hr className="mt-2" />

                                  {/* ============================= */}
                                  {/* BENEFIT ROW */}
                                  {/* ============================= */}
                                  <div className="flex flex-row mt-4 ml-2 items-start justify-between">
                                    <div className="w-full">
                                      <h1 className="text-green-900 mb-2">
                                        Benefit Score & Analysis
                                      </h1>
                                      <div className="text-neutral-900">
                                        {record.benefit_reasoning == null ? (
                                          <div className="loader text-sm"></div>
                                        ) : (
                                          record.benefit_reasoning
                                        )}
                                      </div>
                                    </div>

                                    {/* SCORE BOX → Tooltip Wrapper */}
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <div className="cursor-pointer relative gap-2 grid">
                                          {isLowConfidence && (
                                            <div className="bg-yellow-100 border text-yellow-600 border-yellow-400 items-center justify-center flex  rounded-md py-1">
                                              <AlertTriangle
                                                size={14}
                                                className=" text-yellow-500 "
                                              />
                                            </div>
                                          )}
                                          <div className="bg-green-100 border border-green-600 aspect-square items-center justify-center flex size-10 rounded-md relative">
                                            <h1 className="text-green-600 text-sm font-semibold">
                                              {record.benefit_score == null
                                                ? "N/A"
                                                : String(record.benefit_score).split(".")[0] + "%"}
                                            </h1>
                                          </div>
                                        </div>
                                      </TooltipTrigger>

                                      {/* Tooltip */}
                                      <TooltipContent
                                        className="
                                          max-w-xs p-2 text-sm
                                          bg-neutral-50 text-neutral-700
                                          border border-neutral-300
                                          rounded-md shadow-md
                                          whitespace-normal text-left leading-snug w-auto

                                        "
                                      >
                                        <p className={`font-semibold mb-1 ${confidenceColor}`}>
                                          Confidence: {confidence?.toFixed(2)}
                                        </p>
                                        <p>{firstSentenceReasoning}</p>
                                      </TooltipContent>
                                    </Tooltip>
                                  </div>

                                  <hr className="mt-4" />

                                  {/* ============================= */}
                                  {/* RISK ROW */}
                                  {/* ============================= */}
                                  <div className="flex flex-row ml-2 mt-4 items-start justify-between">
                                    <div className="w-full">
                                      <h1 className="text-red-900 mb-2">Risk Score & Analysis</h1>
                                      <div className="text-neutral-900">
                                        {record.risk_reasoning == null ? (
                                          <div className="loader text-sm"></div>
                                        ) : (
                                          record.risk_reasoning
                                        )}
                                      </div>
                                    </div>

                                    {/* SCORE BOX → Tooltip Wrapper */}
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <div className="cursor-pointer relative grid gap-2">
                                          {isLowConfidence && (
                                            <div className="bg-yellow-100 border text-yellow-600 border-yellow-400 items-center justify-center flex  rounded-md py-1">
                                              <AlertTriangle
                                                size={14}
                                                className=" text-yellow-500 "
                                              />
                                            </div>
                                          )}
                                          <div className="bg-red-100 border border-red-600 aspect-square items-center justify-center flex size-10 rounded-md relative">
                                            <h1 className="text-red-600 text-sm font-semibold">
                                              {record.risk_score == null
                                                ? "N/A"
                                                : String(record.risk_score).split(".")[0] + "%"}
                                            </h1>
                                          </div>
                                        </div>
                                      </TooltipTrigger>

                                      {/* Tooltip */}
                                      <TooltipContent
                                        className="
                                          max-w-xs p-2 text-sm
                                          bg-neutral-50 text-neutral-700
                                          border border-neutral-300
                                          rounded-md shadow-md
                                          whitespace-normal text-left leading-snug w-auto
                                        "
                                      >
                                        <p className={`font-semibold mb-1 ${confidenceColor}`}>
                                          Confidence: {confidence?.toFixed(2)}
                                        </p>
                                        <p>{firstSentenceReasoning}</p>
                                      </TooltipContent>
                                    </Tooltip>
                                  </div>
                                </motion.div>
                              </AccordionContent>
                            </AccordionItem>
                          );
                        });
                    })()}
                  </Accordion>
                </div>

                {/* Clear All History */}
                {typeof window !== "undefined" &&
                  (() => {
                    const data = JSON.parse(localStorage.getItem("echoguard_recordings") || "[]");
                    if (data.length === 0) return null;

                    return (
                      <Button
                        className="absolute bottom-2 right-2 bg-white text-red-500 border-red-500 border hover:bg-red-500 hover:text-red-50"
                        size="sm"
                        onClick={() => {
                          localStorage.removeItem("echoguard_recordings");
                          window.location.reload();
                        }}
                      >
                        Clear All History
                      </Button>
                    );
                  })()}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default DashboardDialog;
