import React from "react";
import {
  HvHeader,
  HvHeaderBrand,
  HvHeaderNavigation,
} from "@hitachivantara/uikit-react-core";
import HitachiLogo from "./HitachiLogo";

const Header = () => {
  const navigationData = [
    {
      id: "1",
      label: "Overview",
    },
  ];
  const selected = "1";

  return (
    <HvHeader position="relative">
      <HvHeaderBrand logo={<HitachiLogo />} name="Lumada App" />

      <HvHeaderNavigation
        data={navigationData}
        selected={selected}
        onClick={() => {}}
      />
    </HvHeader>
  );
};

export default Header;
