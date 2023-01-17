import React, { Suspense } from "react";
import FileUploadForm from "./pages/FileUploadForm/FileUploadForm";
import { configureI18n } from "./lib/i18n";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import "./App.css";
import { Container } from "@mui/material";
import Dashboard from "./pages/Dashboard";
import Header from "./components/Header/Header";
import useStyles from "./styles";
import { NotificationProvider } from "./lib/context/NotificationContext";

const theme = createTheme({});

function App() {
  configureI18n("/");
  const classes = useStyles();

  return (
    <Suspense fallback>
      <ThemeProvider theme={theme}>
        <NotificationProvider>
          <div className="App">
            <Header />
            <Container maxWidth="lg" className={classes.container}>
              <FileUploadForm />
              <Dashboard />
            </Container>
          </div>
        </NotificationProvider>
      </ThemeProvider>
    </Suspense>
  );
}

export default App;
