import React from "react";
import { HvContainer } from "@hitachivantara/uikit-react-core";

const BottomPart = () => {
    const header = "Cenas";
    const title = "Header";
    const content = "Footer";
    return (
        <HvContainer>
            <div>{title}</div>
            <hr/>
            <table>
                <tr>{header}</tr>
                <td>Cenas 2</td>
                <td>Cenas 2</td>
            </table>
            <hr></hr>
            <div>{content}</div>
        </HvContainer>
    );
};

export default BottomPart;
