"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, Variants, Transition } from "framer-motion";
import { Button } from "./ui/button";
import { CircleX, Settings } from "lucide-react";

const DashboardDialog = () => {
  const [isOpen, setIsOpen] = useState(false);
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
    setIsOpen(!isOpen);
  };

  const closeModal = () => {
    setIsOpen(false);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        closeModal();
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
      <Button onClick={toggleModal} size="icon" variant="outline" className=" border-neutral-200">
        <Settings size={15} />
      </Button>

      {/* Main modal */}
      <AnimatePresence mode="wait">
        {isOpen && (
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
              <div className="h-full w-full bg-white rounded-lg shadow-sm dark:bg-gray-700">
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
                <div className="p-4 md:p-5 space-y-4">this is the body of the modal</div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default DashboardDialog;
