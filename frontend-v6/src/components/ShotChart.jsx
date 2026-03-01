import React from 'react';
import { motion } from 'framer-motion';

const ShotChart = ({ shots, title = "Shot Chart" }) => {
    // FIBA Court Dimensions Scaled to SVG (1 meter = 10 units)
    // Court: 28m x 15m -> 280 x 150 viewBox
    const SVG_W = 280;
    const SVG_H = 150;

    // Basket Centers (FIBA basket is ~1.575m from baseline)
    const LEFT_HOOP_X = 15.75;
    const HOOP_Y = 75; // center of 150

    const RIGHT_HOOP_X = 264.25;

    return (
        <div className="w-full h-full flex flex-col items-center">
            <h3 className="text-xl font-semibold mb-4 text-slate-200">{title}</h3>
            <div className="w-full aspect-[28/15] bg-[#cc7a40]/10 border-2 border-slate-700 rounded-xl overflow-hidden relative shadow-inner flex items-center justify-center p-2">
                <svg
                    viewBox={`0 0 ${SVG_W} ${SVG_H}`}
                    className="w-full h-full drop-shadow-md"
                    vectorEffect="non-scaling-stroke"
                >
                    {/* Court Background */}
                    <rect x="0" y="0" width={SVG_W} height={SVG_H} fill="#d97736" stroke="#f8fafc" strokeWidth="2" opacity="0.1" />

                    {/* Half Court Line */}
                    <line x1={SVG_W / 2} y1="0" x2={SVG_W / 2} y2={SVG_H} stroke="#f8fafc" strokeWidth="1" opacity="0.5" />
                    {/* Center Circle */}
                    <circle cx={SVG_W / 2} cy={SVG_H / 2} r="18" fill="none" stroke="#f8fafc" strokeWidth="1" opacity="0.5" />

                    {/* Left Paint (FIBA 5.8m x 4.9m) -> 58 x 49 */}
                    <rect x="0" y={HOOP_Y - 24.5} width="58" height="49" fill="none" stroke="#f8fafc" strokeWidth="1" opacity="0.5" />
                    {/* Right Paint */}
                    <rect x={SVG_W - 58} y={HOOP_Y - 24.5} width="58" height="49" fill="none" stroke="#f8fafc" strokeWidth="1" opacity="0.5" />

                    {/* Left 3PT Arc (radius 6.75m -> 67.5) from hoop center */}
                    {/* Rough approximation with svg path */}
                    <path d={`M 0,${HOOP_Y - 67.5} L 29.9,${HOOP_Y - 67.5} A 67.5 67.5 0 0 1 29.9,${HOOP_Y + 67.5} L 0,${HOOP_Y + 67.5}`} fill="none" stroke="#f8fafc" strokeWidth="1" opacity="0.5" />

                    {/* Right 3PT Arc */}
                    <path d={`M ${SVG_W},${HOOP_Y - 67.5} L ${SVG_W - 29.9},${HOOP_Y - 67.5} A 67.5 67.5 0 0 0 ${SVG_W - 29.9},${HOOP_Y + 67.5} L ${SVG_W},${HOOP_Y + 67.5}`} fill="none" stroke="#f8fafc" strokeWidth="1" opacity="0.5" />

                    {/* Left Hoop */}
                    <circle cx={LEFT_HOOP_X} cy={HOOP_Y} r="2.25" fill="#f97316" stroke="#ea580c" strokeWidth="1" />
                    {/* Right Hoop */}
                    <circle cx={RIGHT_HOOP_X} cy={HOOP_Y} r="2.25" fill="#f97316" stroke="#ea580c" strokeWidth="1" />


                    {/* Shots Plotting */}
                    {shots && shots.map((shot, idx) => {
                        if (!shot.coord_m) return null;

                        // Map real-world meters to SVG coordinates
                        const [cx, cy] = shot.coord_m;
                        const svgX = cx * 10;
                        const svgY = cy * 10;

                        const isMade = shot.status === 'made';

                        return (
                            <motion.circle
                                key={idx}
                                initial={{ r: 0, opacity: 0 }}
                                animate={{ r: 3, opacity: 0.8 }}
                                whileHover={{ r: 5, opacity: 1 }}
                                transition={{ duration: 0.5, delay: idx * 0.1 }}
                                cx={svgX}
                                cy={svgY}
                                fill={isMade ? "#10b981" : "#ef4444"}
                                stroke={isMade ? "#047857" : "#b91c1c"}
                                strokeWidth="1"
                                className="cursor-pointer"
                            >
                                <title>{`Player ${shot.shooter_id}: ${isMade ? 'Make' : 'Miss'} from ${shot.zone}`}</title>
                            </motion.circle>
                        );
                    })}
                </svg>
            </div>
            <div className="flex gap-6 mt-4 text-xs font-semibold text-slate-400 uppercase tracking-widest">
                <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block drop-shadow-sm"></span> Made
                </div>
                <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-rose-500 inline-block drop-shadow-sm"></span> Missed
                </div>
            </div>
        </div>
    );
};

export default ShotChart;
