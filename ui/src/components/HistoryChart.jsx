import React from 'react';
import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line } from 'recharts';

const chartConfig = {
    sensors: {
        lines: [
            { dataKey: 'temperature', stroke: '#F87171', yAxisId: 'left', name: 'Temp (°C)' },
            { dataKey: 'humidity', stroke: '#60A5FA', yAxisId: 'right', name: 'Humidity (%)' },
            { dataKey: 'co2', stroke: '#4ADE80', yAxisId: 'right', name: 'CO2 (ppm)' },
            { dataKey: 'voc', stroke: '#A78BFA', yAxisId: 'right', name: 'VOC (ppb)' },
        ],
        yAxes: [
            { yAxisId: 'left', stroke: '#F87171' },
            { yAxisId: 'right', orientation: 'right', stroke: '#60A5FA' }
        ]
    },
    globalComfort: {
        lines: [{ dataKey: 'globalComfort', stroke: '#10B981', name: 'Global Comfort' }],
        yAxes: [{ yAxisId: 'left', stroke: '#10B981', domain: [0, 100] }]
    },
    heatIndex: {
        lines: [{ dataKey: 'heatIndex', stroke: '#FBBF24', name: 'Heat Index' }],
        yAxes: [{ yAxisId: 'left', stroke: '#FBBF24' }]
    },
    iaqIndex: {
        lines: [{ dataKey: 'iaqIndex', stroke: '#38BDF8', name: 'IAQ Index' }],
        yAxes: [{ yAxisId: 'left', stroke: '#38BDF8', domain: [0, 100] }]
    },
};

const HistoryChart = ({ data, view }) => {
    const currentConfig = chartConfig[view] || chartConfig.sensors;
    return (
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" tick={{ fontSize: 10 }} />
                {currentConfig.yAxes.map(axis => <YAxis key={axis.yAxisId} {...axis} tick={{ fontSize: 10 }} />)}
                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '0.5rem' }} labelStyle={{ color: '#9CA3AF' }} />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                {currentConfig.lines.map(line => <Line key={line.dataKey} {...line} type="monotone" strokeWidth={2} dot={false} />)}
            </LineChart>
        </ResponsiveContainer>
    );
};

export default HistoryChart;
