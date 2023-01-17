import { makeStyles } from "@mui/styles";

const styles = makeStyles(() => ({
  container: {
    minHeight: "30vh",
    borderBottom: "1px solid #ccc",
  },
  mainContainer: {
    // justifyContent: "space-around",
  },
  chooseContainer: {
    textAlign: "right",
  },
  minHeight: {
    minHeight: 36,
  },
  itemContainer: {
    padding: 5,
    textAlign: "center",
    lineHeight: 2.1,
    minHeight: 36,
    backgroundColor: "white",
    boxShadow:
      "0px 2px 1px -1px rgb(0 0 0 / 20%), 0px 1px 1px 0px rgb(0 0 0 / 14%), 0px 1px 3px 0px rgb(0 0 0 / 12%)",
  },
  baseContainer: {
    flexWrap: "nowrap",
    margin: 0,
    width: "100%",
  },
  itemIconContainer: {
    marginRight: 10,
    display: "flex",
    alignItems: "center",
    flexWrap: "wrap",
  },
  itemTextContainer: {
    overflow: "hidden",
    whiteSpace: "nowrap",
    textOverflow: "ellipsis",
    margin: "auto",
  },
  loadingBar: {
    height: 26,
  },
  recommendationContainer: {
    textAlign: "left",
    minHeight: 80,
    display: "flex",
    flexDirection: "column",
    textColor: "#fff",
  },
  actionContainer: {
    display: "flex",
    justifyContent: "flex-end",
  }
}));

export default styles;
