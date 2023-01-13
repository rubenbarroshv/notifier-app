import React from "react";

const useLocalStorage = (key, value) => {
  const [stateValue, setStateValue] = React.useState(() => {
    if (typeof window === "undefined") return value;

    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : value;
    } catch (error) {
      return value;
    }
  });

  const setStoreValue = (newValue) => {
    try {
      setStateValue(newValue);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(key, JSON.stringify(newValue));
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.log(error);
    }
  };

  return [stateValue, setStoreValue];
};

export default useLocalStorage;
