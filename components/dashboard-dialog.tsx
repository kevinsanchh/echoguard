"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, Variants, Transition } from "framer-motion";
import { Button } from "./ui/button";
import { AudioLines, CircleX, Settings } from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import RecordingExpandedDialog from "./recording-expanded-dialog";
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
    initial: {
      opacity: 0, // Start fully transparent (the "fade")
      scale: 0.9, // Start slightly smaller (the "stuff")
    },
    animate: {
      opacity: 1, // Animate to fully opaque
      scale: 1, // Animate to normal size
    },
  };

  const defaultTransition: Transition = {
    ease: "easeOut", // Starts fast and slows down at theend
    duration: 0.2, // The animation takes 0.2 seconds
  };

  const backdropVariants: Variants = {
    initial: {
      opacity: 0,
    },
    animate: {
      opacity: 1,
    },
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
      // Add event listener when the modal is open
      document.addEventListener("mousedown", handleClickOutside);
    }

    // Cleanup the event listener on component unmount or when modal is closed
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  return (
    <>
      {/* Modal toggle */}
      {!externalToggleModal && (
        <Button onClick={toggleModal} size="icon" variant="outline" className=" border-neutral-200">
          <AudioLines size={15} />
        </Button>
      )}

      {/* Main modal */}
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
            className="overflow-y-auto overflow-x-hidden fixed top-0 right-0 left-0 z-50 flex justify-center items-center w-full md:inset-0 h-[calc(100%)] max-h-full bg-black/20 backdrop-blur-xs"
          >
            {/* The ref is attached to the modal content's wrapper, not the backdrop */}
            <motion.div
              key="terms-modal"
              ref={modalRef}
              variants={defaultVariants}
              transition={defaultTransition}
              initial="initial"
              animate="animate"
              exit="initial"
              className="relative p-4 w-full h-full max-h-[90%] max-w-lg aspect-3/4"
            >
              {/* Modal content */}
              <div className="h-full w-full relative bg-white rounded-lg shadow-sm dark:bg-gray-700">
                {/* Modal Header */}
                <div className="p-4 md:p-5 flex justify-between items-center border-b border-neutral-100 ">
                  <h1 className="font-semibold text-xl">Dashboard</h1>
                  <Button
                    onClick={closeModal}
                    size="icon"
                    variant="outline"
                    className=" border-neutral-200 rounded-full"
                  >
                    <CircleX size={15} className="text-neutral-400" />
                  </Button>
                </div>

                {/* Modal body */}
                <div className="p-4 md:p-5 space-y-4 overflow-y-scroll">
                  <h3 className="font-semibold text-lg mb-2">Saved Recordings</h3>

                  {/* Designing new list item WIP */}

                  <Accordion
                    type="single"
                    collapsible
                    className="flex flex-col gap-2  overflow-y-scroll max-h-[36.8rem]"
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
                        .map((record: any, idx: number) => (
                          <AccordionItem
                            key={record.id}
                            value={`item-${idx}`}
                            className="border rounded-md cursor-pointer active:scale-[0.97] duration-100 bg-white"
                          >
                            <AccordionTrigger>
                              <div className="w-full rounded-md p-3 bg-background/80 backdrop-blur-sm ">
                                <h1 className="font-semibold mb-3 border-b pb-2">
                                  Recording {format(new Date(record.date), "MMM. dd, yyyy h:mm a")}
                                </h1>
                                <div className="hidden flex-row gap-2 my-4">
                                  <div className="px-1 rounded-sm font-medium text-xs bg-green-100 border-green-600 text-green-600 border">
                                    Benefit Score: 62%
                                  </div>
                                  <div className="px-1 rounded-sm font-medium text-xs bg-red-100 border-red-600 text-red-600 border">
                                    Risk Score : 13%
                                  </div>
                                </div>
                                <div className="grid grid-cols-3 gap-2">
                                  {record.detections.map((det: any, i: number) => (
                                    <div
                                      key={i}
                                      className="border bg-neutral-50 border-neutral-200 rounded-md p-2"
                                    >
                                      <h1 className="text-md text-neutral-600">{det.label}</h1>
                                      <h1 className="font-semibold text-right text-xl">
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
                                {/* <p>{record.benefitReasoning}heere is some contetn</p>
                                <p>{record.riskReasoning} here is some content</p> */}
                                <hr className="mt-2" />
                                <div className="flex flex-row mt-4">
                                  <div className="w-full">
                                    <h1 className="text-neutral-600 mb-2">
                                      Benefit Score & Analysis
                                    </h1>
                                    <p className="text-neutral-900">{record.benefitReasoning}</p>
                                    <p className="text-neutral-900">
                                      Officia ullamco in consectetur exercitation esse exercitation
                                      laborum eu et adipisicing cillum do eiusmod voluptate ex. Ex
                                      deserunt sunt amet enim. Occaecat quis adipisicing do quis ex
                                    </p>
                                  </div>
                                  <div className="bg-green-100 border border-green-600 aspect-square items-center justify-center flex size-10">
                                    <h1 className="text-green-600 text-sm font-semibold">34%</h1>
                                  </div>
                                </div>
                                <hr className="mt-4" />
                                <div className="flex flex-row mt-4">
                                  <div className="w-full">
                                    <h1 className="text-neutral-600 mb-2">Risk Score & Analysis</h1>
                                    <p className="text-neutral-900">{record.riskReasoning}</p>
                                    <p className="text-neutral-900">
                                      Officia ullamco in consectetur exercitation esse exercitation
                                      laborum eu et adipisicing cillum do eiusmod voluptate ex. Ex
                                      deserunt sunt amet enim. Occaecat quis adipisicing do quis ex
                                    </p>
                                  </div>
                                  <div className="bg-red-100 border border-red-600 aspect-square items-center justify-center flex size-10">
                                    <h1 className="text-red-600 text-sm font-semibold">34%</h1>
                                  </div>
                                </div>
                              </motion.div>
                            </AccordionContent>
                          </AccordionItem>
                        ));
                    })()}
                  </Accordion>
                </div>
                {typeof window !== "undefined" &&
                  (() => {
                    const data = JSON.parse(localStorage.getItem("echoguard_recordings") || "[]");
                    if (data.length === 0) return null; // hide button if no recordings

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
