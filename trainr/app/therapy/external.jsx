import React from "react";
import Container from "@mui/material/Container";
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

export default function ExternalRotation() {
  console.log("Loaded");
  return (
    <div
      style={{
        position: "absolute",
        height: "100%",
        width: "100%",
        backgroundColor: "lightBlue",
      }}
    >
      <h1
        style={{
          margin: "10px auto",
          padding: "4px 8px", // Adjust padding to control spacing around text
          textAlign: "center",
          backgroundColor: "white",
          display: "inline-block", // Keeps the width only as wide as the text
          borderRadius: "4px", // Optional: Adds slightly rounded corners for a neater look
        }}
      >
        External Rotation
      </h1>

      <Container
        style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-evenly",
          alignItems: "center",
          width: "100%",
        }}
      >
        <div style={{ padding: "10px", border: "2px solid black" }}>
          <h2 style={{ marginTop: "0px" }}>Follow this Exercise!</h2>
          <img
            src={
              "/assets/image/Standing_External_Rotation_with_Resistance_Band.gif"
            }
            alt="Standing External Rotation"
            style={{
              width: "400px",
              height: "200px",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
        </div>
        <div style={{ padding: "10px", border: "2px solid black" }}>
          <h2 style={{ marginLeft: "20px" }}>YOU!</h2>
          <img
            // src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAJQA+QMBIgACEQEDEQH/xAAWAAEBAQAAAAAAAAAAAAAAAAAAAQf/xAAYEAEBAAMAAAAAAAAAAAAAAAAAAREhQf/EABUBAQEAAAAAAAAAAAAAAAAAAAAB/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AxFFQAAAUBBRBBQEIEBQAABQABFBEVAFAAAAAAAAVABFAQVAFRQAQBUiglCgAAKAKAAAAIqCAAAKCKigAAAAAAIoCKAIAAAKAoiACioCKIooAAACCgiAoIoAIoBF0iggAoAIACoAIqEKAAAAKACAAKIooCAoAAAgAAAAAAAKAAAAIoCACAAAAAAAAAACgAAKAAACAAARcAgCgAigAAACKlAAEAAAAAAAAFQBRFFAAABAABUAAFABFAAAAAAQKCAAAAAAAAAACoAoAAAAAAAAKCCoKAAAAACJQAAAAAAAAAAAAAFRQAAF4gAAAAKqAAAAAIJVSgAAAAAAAAAAAoICgiooAAoACooAgKgAgAAAAJVAQBQAQAAAAAAUAAAAAAABQFAAf/9k="
            src="http://localhost:5000/video-feed"
            alt="Video Stream"
            style={{
              width: "400px",
              height: "200px",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
        </div>
      </Container>
      <div style={{ display: "flex", justifyContent: "center" }}>
        <div style={boxStyle}>
          <div>Repetitions</div>
          <div style={textStyle}>3 sets of 8</div>
        </div>
        <div style={boxStyle}>
          <div>Days per week</div>
          <div style={textStyle}>3</div>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginTop: "30px",
        }}
      >
        <div
          style={{
            border: "2px solid black",
            borderRadius: "4px",
            width: "500px",
            padding: "10px",
            textAlign: "left",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <h3
              style={{
                margin: "0px",
                padding: "4px 8px",
                backgroundColor: "white",
                display: "inline-block",
                borderRadius: "5px",
              }}
            >
              {" "}
              Procedure
            </h3>
          </div>
          <ol>
            <li style={{ fontSize: "20px" }}>
              Make a 3-foot-long loop with the elastic band and tie the ends
              together. Attach the loop to a doorknob or other stable object.
            </li>
            <li style={{ fontSize: "20px" }}>
              Stand holding the band with your elbow bent and at your side, as
              shown in the start position.
            </li>
            <li style={{ fontSize: "20px" }}>
              Keeping your elbow close to your side, slowly rotate your arm
              outward.
            </li>
            <li style={{ fontSize: "20px" }}>
              Slowly return to the start position and repeat.
            </li>
          </ol>
        </div>
      </div>

      <div style={{ position: "absolute", backgroundColor: "black" }}></div>
    </div>
  );
}
