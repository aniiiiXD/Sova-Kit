#!/usr/bin/env node
import { exitLikePython, launchPython } from "../lib/python-bridge.js";
exitLikePython(await launchPython("switchit.mcp_server"));
