"use client";

import React from "react";
import Link from "next/link";
import Container from "@mui/material/Container";
import { useRouter } from "next/navigation";

const boxStyle = {
  backgroundColor: "#2F4F4F",
  color: "white",
  padding: "10px",
  textAlign: "center",
  margin: "10px",
  borderRadius: "5px",
  fontFamily: "Arial, sans-serif",
  fontWeight: "bold",
  width: "150px",
};

const textStyle = {
  backgroundColor: "white",
  color: "black",
  padding: "10px",
  fontWeight: "normal",
};

function Page() {
  const router = useRouter();

  return (
    <div
      style={{
        position: "absolute",
        height: "100%",
        width: "100%",
        backgroundColor: "lightBlue",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <h1
        style={{
          margin: "0", // Remove default margin
          padding: "4px 8px",
          textAlign: "left", // Align text to the left
          backgroundColor: "white",
          borderRadius: "4px",
          position: "absolute", // Allows it to be positioned without affecting the flex layout
          top: "10px", // Position at the top
          left: "10px", // Position at the left
        }}
      >
        Hello TestNAME!
      </h1>

      <div style={{ display: "inline-block" }}>
        <div
          style={{ width: "400px", backgroundColor: "white", display: "flex" }}
        >
          <div
            style={{
              border: "2px solid black",
              padding: "5px",
              textAlign: "center",
              display: "inline-block",
            }}
          >
            <h3
              style={{
                margin: "0px",
                padding: "4px 8px",
                backgroundColor: "white",
                display: "inline-block",
                borderRadius: "5px",
              }}
            >
              External Rotations
            </h3>

            <p style={{ margin: "2px 5px 2px 5px" }}>
              Focuses on <i>Infraspinatus, teres minor, posterior deltoid</i>.
              You should feel this stretch in the back of your shoulder and
              upper back.
            </p>
            <img
              src="./assets/images/Standing-Shoulder-Rotation.png" // Use forward slashes for paths
              width="200px"
              height="200px"
              style={{ marginTop: "20px" }} // Apply marginTop correctly within style
              alt="Standing Shoulder Rotation" // Consider adding an alt attribute for accessibility
            />
            <br />
            <Link href="/therapy/external">
              <h3 style={{ cursor: "pointer" }}>TESTING</h3>{" "}
              {/* Ensured it's clickable */}
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Page;
