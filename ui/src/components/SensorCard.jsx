import React from 'react';
import { formatValue } from '../utils/api';

const SensorCard = ({ title, icon, value, unit, decimals = 0 }) => (
    <div className="bg-gray-800 p-3 rounded-lg flex items-center">
        {icon}
        <div className="ml-3">
            <div className="text-gray-400 text-sm">{title}</div>
            <div className="text-white text-xl font-semibold">
                {formatValue(value, decimals)} <span className="text-sm text-gray-400">{unit}</span>
            </div>
        </div>
    </div>
);

export default SensorCard;
