
"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
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
            "radial-gradient(at 20% 25%, hsla(212, 95%, 25%, 0.45) 0px, transparent 50%), radial-gradient(at 80% 20%, hsla(340, 95%, 30%, 0.35) 0px, transparent 50%), radial-gradient(at 50% 80%, hsla(240, 85%, 28%, 0.4) 0px, transparent 50%)",
        }}
      />
      <div className="absolute inset-0 backdrop-blur-[90px]" />

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
            <div className="absolute -inset-[2px] rounded-full bg-white/40 blur-[10px] opacity-70 transition group-hover:opacity-90" />
            <div className="relative flex items-center rounded-full border border-white/15 bg-[#111a2d]/85 px-4 py-2 shadow-2xl backdrop-blur-lg">
              <input
                type="number"
                placeholder="Game ID..."
                value={gameId}
                onChange={(e) => setGameId(e.target.value)}
                className="flex-1 bg-transparent px-2 py-3 text-lg text-white placeholder-white/40 focus:outline-none"
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
