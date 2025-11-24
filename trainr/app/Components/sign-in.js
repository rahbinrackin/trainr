"use client"; // Mark as a client component for Firebase authentication and routing

import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  TextField,
  Typography,
  Divider,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import { auth } from "../firebase/config";
import { signInWithEmailAndPassword } from "firebase/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

const Card = styled(Box)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  padding: theme.spacing(4),
  gap: theme.spacing(2),
  backgroundColor: "#fff",
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[5],
  maxWidth: "400px",
  width: "100%",
  margin: "auto",
}));

export default function SignIn() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const email = data.get("email");
    const password = data.get("password");

    try {
      await signInWithEmailAndPassword(auth, email, password);
      setIsAuthenticated(true);
    } catch (error) {
      setError("Login unsuccessful: " + error.message);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      router.push("/therapy"); // Navigate to the therapy page after logging in
    }
  }, [isAuthenticated, router]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Card component="form" onSubmit={handleSubmit}>
        <Typography
          component="h1"
          variant="h4"
          sx={{ color: "black" }}
          gutterBottom
        >
          Sign In
        </Typography>
        <TextField
          fullWidth
          id="email"
          label="Email"
          type="email"
          name="email"
          variant="outlined"
          required
        />
        <TextField
          fullWidth
          id="password"
          label="Password"
          name="password"
          type="password"
          variant="outlined"
          required
        />
        <FormControlLabel
          control={<Checkbox value="remember" />}
          label={<Typography sx={{ color: "black" }}>Remember me</Typography>}
        />
        <Button type="submit" fullWidth variant="contained">
          Sign In
        </Button>
        {error && (
          <Typography color="error" sx={{ textAlign: "center", mt: 2 }}>
            {error}
          </Typography>
        )}
        <Typography sx={{ textAlign: "center", mt: 2 }}>
          Don&apos;t have an account?{" "}
        </Typography>
        <Divider sx={{ my: 2 }} />
      </Card>
    </Box>
  );
}
