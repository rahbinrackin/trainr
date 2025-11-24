"use client";

import React, { useEffect, useState } from "react";

export default function Home() {
  const [error, setError] = useState(null);
  const videoSrc = "http://localhost:5001/video-feed";

  useEffect(() => {
    // Add timestamp to prevent caching
    const img = document.getElementById("openpose-video");
    if (img) {
      img.src = videoSrc + "?t=" + Date.now();
      
      img.onerror = () => {
        setError("Unable to connect to OpenPose server. Make sure the server is running on port 5001.");
      };
      
      img.onload = () => {
        setError(null);
      };
    }
  }, []);

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
        flexDirection: "column",
      }}
    >
      <div style={{ padding: "10px", border: "2px solid black", backgroundColor: "white" }}>
        <h2 style={{ marginTop: "0px", textAlign: "center" }}>OpenPose Video Feed</h2>
        {error && (
          <div style={{ 
            color: "red", 
            textAlign: "center", 
            padding: "10px",
            backgroundColor: "#ffebee",
            borderRadius: "4px",
            marginBottom: "10px"
          }}>
            {error}
          </div>
        )}
        <img
          id="openpose-video"
          src={videoSrc}
          alt="OpenPose Video Stream"
          style={{
            width: "800px",
            height: "600px",
            marginLeft: "auto",
            marginRight: "auto",
            display: "block",
            border: "1px solid #ccc",
          }}
        />
      </div>
    </div>
  );
}