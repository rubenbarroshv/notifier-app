import React from "react";
import { HvDonutchart } from "@hitachivantara/uikit-react-viz";
import { HvContainer, HvGrid } from "@hitachivantara/uikit-react-core";
import { useTheme } from "@mui/material/styles";
import { NotificationContext } from "../lib/context/NotificationContext";
import { percentage } from "../lib/utils/utils";

const Dashboard = () => {
  const { notifications } = React.useContext(NotificationContext);
  const theme = useTheme();

  const normal =
    notifications?.reduce((acc, n) => acc + (n.state === "normal"), 0) || 0;
  const defect =
    notifications?.reduce((acc, n) => acc + (n.state === "defect"), 0) || 0;

  const normalValues = notifications?.reduce(
    (acc, n) => {
      if (n.state === "defect") return acc;
      if (n.score < 0.33) return { ...acc, 0: acc[0] + 1 };
      if (n.score < 0.66) return { ...acc, 1: acc[1] + 1 };
      if (n.score < 0.9) return { ...acc, 2: acc[2] + 1 };
      return { ...acc, 3: acc[3] + 1 };
    },
    { 0: 0, 1: 0, 2: 0, 3: 0 }
  );
  const defectValues = notifications?.reduce(
    (acc, n) => {
      if (n.state === "normal") return acc;
      if (n.score < 0.33) return { ...acc, 0: acc[0] + 1 };
      if (n.score < 0.66) return { ...acc, 1: acc[1] + 1 };
      if (n.score < 0.9) return { ...acc, 2: acc[2] + 1 };
      return { ...acc, 3: acc[3] + 1 };
    },
    { 0: 0, 1: 0, 2: 0, 3: 0 }
  );

  const green = theme.hv.palette.semantic.sema1;
  const yellow = theme.hv.viz.palette.categorical.cviz7;
  const orange = theme.hv.palette.semantic.sema3;
  const red = theme.hv.palette.semantic.sema4;

  return (
    <HvContainer style={{ marginTop: 50 }}>
      {(!!normal || !!defect) && (
        <HvGrid container>
          <HvGrid item xs={12} justifyContent="space-around">
            <HvDonutchart
              title="Simple Donut"
              subtitle="Server Status Summary"
              data={[
                {
                  values: [normal, defect],
                  labels: ["Normal", "Defect"],
                  name: "Audio Samples",
                  marker: {
                    colors: [green, red],
                  },
                  sort: false,
                },
              ]}
              layout={{
                width: 500,
                height: 400,
                annotations: [
                  {
                    text: `Normal ${percentage(normal / (normal + defect))}%`,
                    showarrow: false,
                  },
                ],
              }}
            />
          </HvGrid>
          <HvGrid item xs={6}>
            <HvDonutchart
              title="Normal Values"
              subtitle="Normal Values Summary"
              data={[
                {
                  values: Object.values(normalValues),
                  labels: ["< 33%", "< 66%", "< 90%", "<= 100%"],
                  name: "Audio Samples",
                  marker: {
                    colors: [red, orange, yellow, green],
                  },
                  sort: false,
                },
              ]}
              layout={{
                width: 500,
                height: 400,
                annotations: [
                  {
                    text: "Probability: Normal values",
                    showarrow: false,
                  },
                ],
              }}
            />
          </HvGrid>
          <HvGrid item xs={6}>
            <HvDonutchart
              title="Defect Values"
              subtitle="Defect Values Summary"
              data={[
                {
                  values: Object.values(defectValues),
                  labels: ["< 33%", "< 66%", "< 90%", "<= 100%"],
                  name: "Audio Samples",
                  marker: {
                    colors: [red, orange, yellow, green],
                  },
                  sort: false
                },
              ]}
              layout={{
                width: 500,
                height: 400,
                annotations: [
                  {
                    text: "Probability: Defect values",
                    showarrow: false,
                  },
                ],
              }}
            />
          </HvGrid>
        </HvGrid>
      )}
    </HvContainer>
  );
};

export default Dashboard;
