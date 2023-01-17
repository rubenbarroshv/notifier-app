import React from "react";
import useLocalStorage from "./useLocalStorage";

const useNotification = () => {
  const [notifications, setNotifications] = React.useState([]);
  const [storage, setStorage] = useLocalStorage("notifier-notifications", []);

  React.useEffect(() => {
    setNotifications(storage);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const addNotification = React.useCallback(
    (notification) => {
      setNotifications((old) => {
        const updatedNotifications = [...old, notification];
        setStorage(updatedNotifications);

        return updatedNotifications;
      });
    },
    [setStorage]
  );

  return {
    notifications,
    addNotification,
  };
};

export default useNotification;
