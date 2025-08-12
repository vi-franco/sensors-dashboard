import React from 'react';
import { ToggleRight } from 'lucide-react';

const ProbBar = ({ value }) => {
    const v = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
    return (
        <div className="w-full bg-gray-700 rounded h-2 overflow-hidden">
            <div className="h-2 bg-indigo-500" style={{ width: `${(v * 100).toFixed(0)}%` }} />
        </div>
    );
};

const Chip = ({ children }) => (
    <span className="text-[11px] px-2 py-0.5 rounded bg-gray-700 text-gray-200 font-mono">{children}</span>
);

const ActuatorTile = ({ name, data, thresholds }) => {
    const isObj = data && typeof data === 'object' && 'state' in data;
    const state = isObj ? !!data.state : !!data;
    const prob  = isObj ? data.prob : null;
    const thr   = thresholds || (isObj ? data.thresholds : null);

    return (
        <div className="bg-gray-800 rounded-md p-3 flex flex-col gap-2">
            <div className="flex items-center justify-between">
                <div className="flex items-center">
                    <span className={`w-2.5 h-2.5 rounded-full mr-2 ${state ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-sm text-gray-200">{name}</span>
                </div>
            </div>

            <div className="flex items-center gap-2">
                <Chip>p={prob == null ? 'N/A' : prob.toFixed(2)}</Chip>
            </div>
            <ProbBar value={prob ?? 0} />

            {thr && (
                <div className="text-[11px] text-gray-400">
                    ON≥{thr.on?.toFixed(2)} · OFF≤{thr.off?.toFixed(2)}
                </div>
            )}
        </div>
    );
};

const ActuatorCard = ({ actuators, available_actuators, predAt, globalHysteresis }) => {
    const enabled = Object.entries(available_actuators || {}).filter(([, en]) => en).map(([k]) => k);
    return (
        <div className="bg-gray-900 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-white font-semibold flex items-center">
                    <ToggleRight size={20} className="mr-2 text-gray-400" />
                    Actuator Status
                </h3>
                {predAt && (
                    <span className="text-xs text-gray-400">
            predicted at {new Date(predAt).toLocaleString('it-IT')}
          </span>
                )}
            </div>

            {enabled.length === 0 ? (
                <p className="text-xs text-gray-500">No enabled actuators.</p>
            ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {enabled.map((name) => (
                        <ActuatorTile
                            key={name}
                            name={name}
                            data={(actuators || {})[name]}
                            thresholds={globalHysteresis?.[name]}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

export default ActuatorCard;
