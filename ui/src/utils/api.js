// Utils: API, formattazioni e helper

export const API_URL =
    (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) ||
    'http://192.168.20.73:5500';

export const formatValue = (value, decimals = 0) => {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    return parseFloat(value).toFixed(decimals);
};

export const fetchJson = async (url, options = {}, signal) => {
    try {
        const res = await fetch(url, { mode: 'cors', ...options, signal });
        if (!res.ok) {
            let body = '';
            try { body = await res.text(); } catch {}
            throw new Error(`HTTP ${res.status} at ${url} — ${body?.slice(0, 200)}`);
        }
        try { return await res.json(); }
        catch {
            let peek = '';
            try { peek = (await res.clone().text()).slice(0, 200); } catch {}
            throw new Error(`Invalid JSON from ${url}. Peek: ${peek}`);
        }
    } catch (e) {
        if (e?.name === 'AbortError') return;
        if (e.message.includes('Failed to fetch')) {
            throw new Error(`Network error (CORS / Mixed Content / PNA?) contacting ${url}`);
        }
        throw e;
    }
};

export const getComfortStatus = (value, thresholds, higherIsBetter = true) => {
    if (value === null || value === undefined) return { text: '-', color: 'text-gray-400' };
    const { good, medium } = thresholds;
    if (higherIsBetter) {
        if (value >= good) return { text: 'Excellent', color: 'text-green-400' };
        if (value >= medium) return { text: 'Good', color: 'text-yellow-400' };
        return { text: 'Poor', color: 'text-red-400' };
    } else {
        if (value <= good) return { text: 'Excellent', color: 'text-green-400' };
        if (value <= medium) return { text: 'Good', color: 'text-yellow-400' };
        return { text: 'Poor', color: 'text-red-400' };
    }
};

export const getPredictionTime = (deviceData) =>
    deviceData?.predictions_meta?.generatedAt
    || deviceData?.predictions?.generatedAt
    || deviceData?.current?.timestamp
    || null;
