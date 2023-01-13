import React from "react";
import useNotification from "../hooks/useNotification";

export const NotificationContext = React.createContext({
  notifications: [],
  addNotification: () => undefined,
});

export const NotificationProvider = ({ children }) => {
  const { notifications, addNotification } = useNotification();

  const value = React.useMemo(
    () => ({ notifications, addNotification }),
    [addNotification, notifications]
  );

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};
