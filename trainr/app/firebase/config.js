// firebase/config.js
import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCMeCkInE6Gy8hM_EGMBNsgUqw8tLg5-_8",
  authDomain: "therapy-ece0f.firebaseapp.com",
  projectId: "therapy-ece0f",
  storageBucket: "therapy-ece0f.appspot.com",
  messagingSenderId: "178362231230",
  appId: "1:178362231230:web:332be91ea741443b8531ef",
};

// Initialize Firebase
const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
const auth = getAuth(app);
const firestore = getFirestore(app);
const storage = getStorage(app);

export { app, auth, firestore, storage };
