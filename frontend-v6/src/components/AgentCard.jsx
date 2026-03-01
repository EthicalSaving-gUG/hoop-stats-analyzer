import React from 'react';
import { motion } from 'framer-motion';

const AGENT_META = {
    shot_clock: { icon: '🕐', title: '24-Sekunden-Uhr', desc: 'Verwaltet die Shot Clock und erkennt Ballbesitzwechsel' },
    game_clock: { icon: '⏱️', title: 'Spielzeiterfassung', desc: 'Steuert Spieluhr, Viertel und Timeouts' },
    scoresheet: { icon: '📋', title: 'Kampfbericht', desc: 'Protokolliert Punkte, Fouls und Auswechslungen' },
    referee: { icon: '🧑‍⚖️', title: 'Schiedsrichter', desc: 'Erkennt Regelverstöße und gibt Warnungen' },
    pose: { icon: '🧍‍♂️', title: 'Biomechanik KI', desc: 'Analysiert Körperhaltung (Release Angle, 3er-Linie) via MediaPipe' },
};

const AgentCard = ({ agentKey, enabled, onToggle, isLive, isActive }) => {
    const meta = AGENT_META[agentKey] || { icon: '❓', title: agentKey, desc: '' };

    return (
        <motion.div
            whileHover={{ scale: 1.02 }}
            className={`glass p-5 cursor-pointer transition-all border-2 ${enabled ? 'border-orange-500/50' : 'border-transparent'
                }`}
            onClick={() => onToggle(agentKey)}
        >
            <div className="flex items-center justify-between mb-3">
                <span className="text-3xl">{meta.icon}</span>
                <label className="relative inline-flex items-center cursor-pointer" onClick={(e) => e.stopPropagation()}>
                    <input
                        type="checkbox"
                        checked={enabled}
                        onChange={() => onToggle(agentKey)}
                        className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-orange-500"></div>
                </label>
            </div>
            <h3 className="text-lg font-semibold mb-1">{meta.title}</h3>
            <p className="text-slate-400 text-sm">{meta.desc}</p>
            {isLive && (
                <div className={`mt-3 text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`}>
                    <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`}></span>
                    {isActive ? 'Aktiv' : 'Pausiert'}
                </div>
            )}
        </motion.div>
    );
};

export { AGENT_META };
export default AgentCard;
