"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "../firebase/config";

const useAuthRedirect = () => {
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (!user) {
        router.push("/");
      }
    });

    const handleBeforeUnload = async () => {
      await signOut(auth);
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      unsubscribe();
    };
  }, [router]);
};

export default useAuthRedirect;
