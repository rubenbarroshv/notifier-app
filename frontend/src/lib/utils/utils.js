export const buildRecommendation = (state, score) => {
  if (state === "normal") {
    if (score < 0.33)
      return [
        "No issue found, but with lower confidence. Consider verifying locally.",
        "schedule",
      ];
    if (score < 0.66)
      return [
        "No issue found, but wit medium confidence. Consider verifying till the next week",
        "schedule",
      ];
    if (score < 0.90)
      return [
        "No issue found. Consider repeating this validation in the next 2 weeks",
      ];
    return ["No issue found, all good!"];
  } else if (state === "defect") {
    if (score < 0.33)
      return ["Probably a future issue here, but no hard maintenance needed."];
    if (score < 0.66)
      return [
        "We detected a possible issue. But with lower priority. Consider repeating the analysis or schedule a maintenance in the next 2 weeks",
        "schedule",
      ];
    if (score < 0.90)
      return [
        "We detected a possible issue. But with medium priority. Consider repeating the analysis or schedule a maintenance for the next week",
        "schedule",
      ];
    return [
      "Alert: problem detected. You need to fix this as soon as possible.",
      "stop",
    ];
  }
};

export const percentage = (
  value
) => {
  if (typeof value === "number") {
    return Math.round(value * 100);
  }
  return "";
};
