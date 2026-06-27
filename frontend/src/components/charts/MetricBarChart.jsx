import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const DEFAULT_COLORS = [
  "#38bdf8",
  "#a78bfa",
  "#34d399",
  "#f59e0b",
  "#fb7185",
];

function truncateLabel(value) {
  const label = String(value ?? "");

  if (label.length <= 18) {
    return label;
  }

  return `${label.slice(0, 16)}...`;
}

function MetricBarChart({
  data,
  bars,
  xKey = "name",
  height = 320,
  emptyMessage = "No chart data available.",
  yDomain,
}) {
  if (!data?.length) {
    return (
      <div className="analytics-empty">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="chart-frame" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{
            top: 12,
            right: 18,
            bottom: 28,
            left: 0,
          }}
        >
          <CartesianGrid
            stroke="rgba(148, 163, 184, 0.16)"
            vertical={false}
          />
          <XAxis
            dataKey={xKey}
            tick={{
              fill: "#94a3b8",
              fontSize: 12,
            }}
            tickFormatter={truncateLabel}
            interval={0}
            minTickGap={8}
          />
          <YAxis
            domain={yDomain}
            tick={{
              fill: "#94a3b8",
              fontSize: 12,
            }}
          />
          <Tooltip
            cursor={{
              fill: "rgba(148, 163, 184, 0.10)",
            }}
            contentStyle={{
              background: "#0f172a",
              border: "1px solid rgba(148, 163, 184, 0.28)",
              borderRadius: 12,
              color: "#e2e8f0",
            }}
            labelStyle={{
              color: "#f8fafc",
            }}
          />
          {bars.length > 1 ? <Legend /> : null}
          {bars.map((bar, index) => (
            <Bar
              key={bar.key}
              dataKey={bar.key}
              name={bar.name}
              fill={bar.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
              radius={[6, 6, 0, 0]}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default MetricBarChart;
