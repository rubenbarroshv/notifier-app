import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useTheme } from "@mui/material/styles";
import axios from "axios";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";
import {
  HvContainer,
  HvButton,
  HvGrid,
  HvLoading,
  HvSnackbar,
} from "@hitachivantara/uikit-react-core";

import useStyles from "./styles";
import clsx from "clsx";
import { buildRecommendation } from "../../lib/utils/utils";
import { NotificationContext } from "../../lib/context/NotificationContext";

function FileUploadForm() {
  const { t } = useTranslation("common");
  const [token, setToken] = useState(null);
  const theme = useTheme();
  const classes = useStyles();
  const { addNotification } = React.useContext(NotificationContext);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadLabel, setUploadLabel] = useState("");
  const [uploadVariant, setUploadVariant] = useState("default");

  const [actionOpen, setActionOpen] = useState(false);
  const [actionLabel, setActionLabel] = useState("");
  const [actionVariant, setActionVariant] = useState("default");

  const [recommendation, setRecommendation] = useState(null);
  const [recommendationType, setRecommendationType] = useState(null);
  const [action, setAction] = useState(false);
  const [actionType, setActionType] = useState(null);

  const handleChange = (e) => {
    setFile(e.target.files[0]);
  };

  const getToken = async () => {
    const result = await axios.post(
      "https://hibi-dev.hitachi-lumada.io/hitachi-solutions/hscp-hitachi-solutions/keycloak/realms/default/protocol/openid-connect/token",
      {
        username: "inference-svc",
        password: "Uth0aeKaeY_eiH4H",
        client_id: "lumada-ml-model-management-inference-svc-client",
        client_secret: "6211fac7-8b89-4af2-9717-7e5d56567bb1",
        grant_type: "password",
      },
      {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      }
    );

    console.log("result", { result });
    setToken(result.data.access_token);
  };

  React.useEffect(() => {
    getToken();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAction(false);
    setRecommendation(null);
    setRecommendationType(null);

    var reader = new FileReader();
    reader.onload = async function () {
      try {
        const text = reader.result;
        const array = text.split("\r\n");

        const json = {
          data: {
            tensor: {
              shape: [84096],
              values: array.map((n) => +n),
            },
          },
        };
        const r = await axios.post(
          "https://hibi-dev.hitachi-lumada.io/hitachi-solutions/lumada-ml-model-management/lumada-ml-model-management-inference-svc-app/api/v0/predictions/bfbec903-370f-43da-bb57-cca619237667/1b050a98-71e3-4174-8185-112794918fd6",
          json,
          {
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${token}`,
            },
          }
        );

        const { class: state, score } = r.data.jsonData.data;

        const [rRec, rAction] = buildRecommendation(state, score);

        setUploadLabel("File uploaded successfully");
        setUploadVariant("success");

        setRecommendation(rRec);
        setRecommendationType(state === "normal" ? "success" : "error");
        setActionType(rAction ?? null);

        setUploadOpen(true);
        setLoading(false);
        addNotification({ state, score });
      } catch (error) {
        setUploadLabel("There was a problem uploading the file", error);
        setUploadVariant("error");
        setUploadOpen(true);

        setLoading(false);
      } finally {
        setFile(null);
      }
    };

    reader.readAsText(file);
  };

  const handleClose = (type) => () => {
    if (type === "upload") setUploadOpen(false);
    if (type === "action") setActionOpen(false);
  };

  const actionHandler = (action) => (event) => {
    if (action === "stop") {
      setActionLabel("Machine is stopping now");
    } else if (action === "schedule") {
      setActionLabel("A maintenance was scheduled for next week");
    }
    if (action !== "ignore") {
      setActionVariant("success");
      setActionOpen(true);
      setAction(true);
    }

    setRecommendation(null);
    setRecommendationType(null);
    setActionType(null);
  };

  const colorSwitcher = (type) =>
    ({
      success: theme.hv.palette.semantic.sema8,
      warning: theme.hv.palette.semantic.sema20,
      error: theme.hv.palette.semantic.sema9,
      default: theme.hv.palette.semantic.sema7,
    }[type]);

  const actionButtonLabel =
    actionType === "stop" ? "Stop machine" : "Schedule a maintenance";

  return (
    <HvContainer className={classes.container}>
      <form onSubmit={handleSubmit}>
        <HvGrid container className={classes.mainContainer}>
          <HvGrid item xs={2} className={classes.chooseContainer}>
            <HvButton
              variant="outlined"
              component="label"
              className={classes.minHeight}
            >
              {t("Choose")}
              {/* .wav */}
              <input
                hidden
                type="file"
                accept="audio/*,.wav"
                onChange={handleChange}
              />
            </HvButton>
          </HvGrid>
          <HvGrid item xs={8}>
            <div className={classes.itemContainer}>
              {loading ? (
                <HvLoading classes={{ loadingBar: classes.loadingBar }} />
              ) : file ? (
                <HvGrid container className={classes.baseContainer}>
                  <div className={classes.itemIconContainer}>
                    <GraphicEqIcon />
                  </div>
                  <div className={classes.itemTextContainer}>{file.name}</div>
                </HvGrid>
              ) : (
                "Please choose a file"
              )}
            </div>
          </HvGrid>
          <HvGrid item xs={2}>
            <HvButton
              category="primary"
              aria-label="Upload"
              className={classes.minHeight}
              onClick={handleSubmit}
              disabled={!file || loading}
            >
              {t("Upload")}
            </HvButton>
          </HvGrid>

          <HvGrid item xs={2} />
          <HvGrid item xs={8}>
            <div
              className={clsx(
                classes.itemContainer,
                classes.recommendationContainer
              )}
              style={{ backgroundColor: colorSwitcher(recommendationType) }}
            >
              {recommendation && (
                <>
                  <div>
                    <p>{recommendation}</p>
                  </div>
                  <div className={classes.actionContainer}>
                    {actionType && (
                      <>
                      <HvButton
                        category="primary"
                        aria-label={actionButtonLabel}
                        className={classes.minHeight}
                        onClick={actionHandler(actionType)}
                        disabled={action}
                      >
                        {actionButtonLabel}
                      </HvButton>
                      <HvButton
                        category="secondary"
                        aria-label="ignore"
                        className={classes.minHeight}
                        onClick={actionHandler("ignore")}
                        style={{ marginLeft: 10 }}
                      >
                        Ignore Recommendation
                      </HvButton>
                      </>
                    )}
                  </div>
                </>
              )}
            </div>
          </HvGrid>
        </HvGrid>
      </form>

      <HvSnackbar
        open={uploadOpen}
        onClose={handleClose("upload")}
        variant={uploadVariant}
        label={uploadLabel}
      />
      <HvSnackbar
        open={actionOpen}
        onClose={handleClose("action")}
        variant={actionVariant}
        label={actionLabel}
      />
    </HvContainer>
  );
}

export default FileUploadForm;
