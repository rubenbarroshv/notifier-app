import React from "react";
import { HvContainer } from "@hitachivantara/uikit-react-core";

const BottomPart = () => {
    const title = "Header";
    const content = "Footer";
    return (
        <HvContainer>
            <div>{title}</div>
            <div>BottomPart</div>
            <div>{content}</div>
        </HvContainer>
    );
};

export default BottomPart;
