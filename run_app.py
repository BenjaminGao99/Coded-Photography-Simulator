#!/usr/bin/env python

#app launcher


import tkinter as tk
from coded_exposure_app import CodedExposureApp

def main():
    root = tk.Tk()
    root.title("Coded Exposure Photography Tool")
    app = CodedExposureApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 