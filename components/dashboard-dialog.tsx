"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, Variants, Transition } from "framer-motion";
import { Button } from "./ui/button";
import { CircleX, Settings } from "lucide-react";

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
          <Settings size={15} />
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
              className="relative p-4 w-full h-full max-h-[90%] max-w-7xl aspect-3/4"
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
                <div className="p-4 md:p-5 space-y-4 ">
                  <h3 className="font-semibold text-lg mb-2">Saved Recordings</h3>
                  <ul className="space-y-2 max-h-[70vh] overflow-y-auto">
                    {(() => {
                      if (typeof window === "undefined") return null;
                      const data = JSON.parse(localStorage.getItem("echoguard_recordings") || "[]");
                      if (data.length === 0)
                        return <li className="text-muted-foreground">No recordings yet</li>;
                      return data
                        .slice()
                        .reverse()
                        .map((record: any) => (
                          <li
                            key={record.id}
                            className="border border-border rounded-md p-3 bg-background/80 backdrop-blur-sm"
                          >
                            <div className="font-medium text-xs text-muted-foreground mb-1">
                              {record.date}
                            </div>
                            <ul className="ml-2 list-disc">
                              {record.detections.map((det: any, i: number) => (
                                <li key={i} className="ml-4">
                                  <span className="font-semibold">{det.label}</span>{" "}
                                  <span className="text-muted-foreground text-xs">
                                    ({det.confidence}%)
                                  </span>
                                </li>
                              ))}
                            </ul>
                          </li>
                        ));
                    })()}
                  </ul>
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
