"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();
  const [gameId, setGameId] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!gameId) return;
    router.push(`/dashboard?gameId=${gameId}`);
  };

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-[#0d1117] text-gray-100">
      <div className="absolute inset-0 bg-[#0d1117]" />
      <div
        className="absolute inset-0 opacity-[0.95]"
        style={{
          backgroundImage:
            "radial-gradient(at 20% 25%, hsla(212, 95%, 22%, 0.28) 0px, transparent 52%), radial-gradient(at 80% 20%, hsla(340, 95%, 26%, 0.24) 0px, transparent 52%), radial-gradient(at 50% 78%, hsla(240, 85%, 24%, 0.3) 0px, transparent 55%)",
        }}
      />
      <div className="absolute inset-0 backdrop-blur-[90px] opacity-80" />

      <section className="relative z-10 flex w-full max-w-4xl flex-col items-center gap-8 px-6 text-center">
        <div>
          <h1 className="font-display text-5xl font-semibold tracking-tight text-white sm:text-6xl">
            moonshot
          </h1>
          <p className="mt-2 text-base text-white/70 sm:text-lg">
            Enter Game ID
          </p>
        </div>

        <form onSubmit={handleSubmit} className="w-full max-w-md">
          <div className="relative group">
            <div className="relative flex items-center rounded-full border border-white/15 bg-[#111a2d]/85 px-4 py-2 shadow-2xl backdrop-blur-lg">
              <input
                type="number"
                placeholder="Game ID..."
                value={gameId}
                onChange={(e) => setGameId(e.target.value)}
                className="flex-1 bg-transparent px-2 py-3 text-lg text-white placeholder-white/40 focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
              <button
                type="submit"
                className="ml-3 flex h-12 w-16 items-center justify-center text-white transition hover:scale-110 focus:outline-none focus:ring-1 focus:ring-white/60"
                aria-label="Submit Game ID"
              >
                <svg
                  viewBox="0 0 32 32"
                  className="h-10 w-10 drop-shadow-[0_0_18px_rgba(255,255,255,0.35)]"
                  fill="none"
                >
                  <path
                    d="M6 16h14"
                    stroke="currentColor"
                    strokeWidth="3"
                    strokeLinecap="round"
                  />
                  <path
                    d="M15 9.5L24 16l-9 6.5"
                    stroke="currentColor"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
          </div>
        </form>
      </section>
    </main>
  );
}

