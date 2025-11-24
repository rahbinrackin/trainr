"use client"; // Add this at the top to mark as a client component

import React, { useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  TextField,
  Typography,
  Stack,
  Divider,
  Link as MuiLink,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth } from "../firebase/config";
import Link from "next/link"; // Import Link from next/link

// Shared Card styling
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

export default function SignUp() {
  const [emailError, setEmailError] = useState(false);
  const [passwordError, setPasswordError] = useState(false);
  const [isSignedUp, setIsSignedUp] = useState(false);
  const [error, setError] = useState("");

  const validateInputs = () => {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    if (!/\S+@\S+\.\S+/.test(email)) {
      setEmailError(true);
      return false;
    } else {
      setEmailError(false);
    }

    if (password.length < 6) {
      setPasswordError(true);
      return false;
    } else {
      setPasswordError(false);
    }

    return true;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (validateInputs()) {
      const email = document.getElementById("email").value;
      const password = document.getElementById("password").value;
      try {
        await createUserWithEmailAndPassword(auth, email, password);
        setIsSignedUp(true); // Update state on successful sign-up
      } catch (error) {
        setError("Sign-up unsuccessful: " + error.message);
      }
    }
  };

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
          Sign Up
        </Typography>
        <TextField
          fullWidth
          id="name"
          label="Full Name"
          variant="outlined"
          required
        />
        <TextField
          fullWidth
          id="email"
          label="Email"
          variant="outlined"
          error={emailError}
          helperText={emailError ? "Invalid email" : ""}
          required
        />
        <TextField
          fullWidth
          id="password"
          label="Password"
          variant="outlined"
          type="password"
          error={passwordError}
          helperText={
            passwordError ? "Password must be at least 6 characters" : ""
          }
          required
        />
        <FormControlLabel
          control={<Checkbox />}
          label={<Typography sx={{ color: "black" }}>Remember Me</Typography>}
        />
        <Button type="submit" fullWidth variant="contained">
          Sign Up
        </Button>
        {isSignedUp && (
          <Typography sx={{ textAlign: "center", mt: 2 }}>
            Successfully signed up! Go to{" "}
            <Link href="/therapy" passHref>
              <MuiLink variant="body2" underline="hover">
                Therapy Session
              </MuiLink>
            </Link>
          </Typography>
        )}
        {error && (
          <Typography color="error" sx={{ textAlign: "center", mt: 2 }}>
            {error}
          </Typography>
        )}
        <Typography sx={{ textAlign: "center", mt: 2 }}>
          Already have an account? <Link href="/sign-in" passHref></Link>
        </Typography>
        <Divider sx={{ my: 2 }} />
      </Card>
    </Box>
  );
}
