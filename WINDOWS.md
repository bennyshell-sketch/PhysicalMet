# Running Dive on Windows

Written for someone who has never used a terminal. Fifteen minutes, most of it
waiting.

---

## Step 1 — Install Python

Dive needs Python. It is free and takes a couple of minutes.

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **Download Python** button
3. Run the file that downloads

**On the first screen of the installer, tick the box that says
"Add python.exe to PATH".** It is at the bottom and it is easy to miss.

> This single checkbox is the most common reason this does not work. If you
> miss it, Windows cannot find Python afterwards and Dive will tell you it is
> not installed. If that happens, run the installer again and tick it.

Then click **Install Now** and wait for it to finish.

---

## Step 2 — Download Dive

You do not need Git. You **do** need to be signed in to GitHub, because the
repository is private — it is yours, and nobody else can see it.

1. Go to **https://github.com** and sign in.

2. Open **https://github.com/bennyshell-sketch/PhysicalMet**

3. Near the top left there is a button showing the current branch, probably
   **main**. Click it and choose **`claude/exciting-ride-s5j9tk`** from the
   list. This is the branch with everything on it.

4. Click the green **Code** button → **Download ZIP**

5. Find the ZIP in your **Downloads** folder. It will be called something like
   `PhysicalMet-claude-exciting-ride-s5j9tk.zip`

6. **Right-click it → Extract All… → Extract**

7. Move the extracted folder somewhere simple, like `C:\Dive`

> If you are signed in, this direct link also works and skips steps 2–4:
> https://github.com/bennyshell-sketch/PhysicalMet/archive/refs/heads/claude/exciting-ride-s5j9tk.zip
>
> If it gives you a "404 — page not found", you are not signed in. Sign in and
> try again. A private repository looks like it does not exist to anyone who is
> not logged in.

> **Move it out of Downloads and out of OneDrive.** OneDrive syncs files while
> the program is using them, which causes odd failures. A plain folder on your
> C: drive is best.

---

## Step 3 — Run it

Open the folder. You will see files including `START-DIVE-windows.bat`.

**Double-click `START-DIVE-windows.bat`.**

A black window opens. On the first run it says *"First run — setting up"* and
takes a few minutes while it downloads what it needs. **Leave the window open.**

When it is ready, your web browser opens automatically at
**http://localhost:8501** and Dive is running.

That is it. Next time it starts in a few seconds.

---

## While it is running

- **Leave the black window open.** Closing it stops Dive.
- To stop it: click the black window and press **Ctrl+C**, or just close it.
- To start it again: double-click `START-DIVE-windows.bat`.
- If the browser does not open on its own, type `localhost:8501` into your
  address bar.

---

## If Windows warns you

You may see **"Windows protected your PC"** with a blue box.

This appears for any file downloaded from the internet that Microsoft has not
seen before. Click **More info**, then **Run anyway**.

Your antivirus may also ask. It is a text file of commands — you can open
`START-DIVE-windows.bat` in Notepad and read exactly what it does.

---

## If something goes wrong

**Double-click `FIX-DIVE-windows.bat`.**

It rebuilds everything from scratch, checks each step, and tells you in plain
words what is wrong. If it cannot fix it, it opens `dive-log.txt` with the full
details — send that file on.

It is safe to run as many times as you like. It only deletes the `.venv`
folder, which it rebuilds. Your data and settings are untouched.

### What it checks, in order

1. Is Python installed and on PATH
2. Rebuilds the environment from scratch
3. Installs every dependency
4. Imports them all
5. Generates the data and loads it
6. Checks whether port 8501 is already taken
7. Starts the app

The first step that fails is the problem, and the log says which.

### The specific messages

## If something goes wrong (details)

The window now stays open and tells you what happened. The common ones:

**"Python was not found"**
The PATH checkbox in Step 1 was missed. Run the Python installer again, tick
**"Add python.exe to PATH"**, then try again.

**"Python 3.9 was found, but Dive needs 3.10 or newer"**
You have an old Python. Install a current one from python.org.

**"Could not create the environment"**
The folder is somewhere Windows restricts. Move it to `C:\Dive` and try again.

**"Installing dependencies failed"**
Usually no internet, or a work firewall blocking downloads. On a work laptop,
try a home network.

**The window flashes and disappears**
You have an old copy of `START-DIVE-windows.bat`. Download the ZIP again — the current version
always stays open.

**The download link gives "404 — page not found"**
You are not signed in to GitHub. The repository is private, so it looks like it
does not exist until you log in. Sign in at github.com and try again.

**"Port 8501 is already in use"**
Dive is already running in another window. Look for it, or restart your
computer.

---

## Checking it installed correctly

Open the folder, hold **Shift**, right-click an empty area, choose
**Open PowerShell window here** or **Open in Terminal**, then type:

```
.\START-DIVE-windows.bat check
```

It runs 17 validation checks and 208 tests. All passing means the installation
is sound.

---

## What to do first

Read **QUICKSTART.md** in the same folder — it walks through each screen.

The short version: open **Data** to see the cohort, then **Individual
prediction**, enter anything and press Predict. It will refuse to give you a
decline risk and explain why. That is the system working correctly, not a bug.
