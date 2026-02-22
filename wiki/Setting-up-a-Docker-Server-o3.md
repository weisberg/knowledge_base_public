# Setting up a Docker Server (o3)
## Executive overview (read this once)

1. **Harden and prepare Windows 11 first.**  Patch, set a static IP, enable virtualization, install OpenSSH and a hardware-level UPS/SMART monitor.
2. **Use Docker Desktop’s WSL 2 backend**—it gives you native-speed Linux containers on Windows with almost zero friction .
3. **Pin all app data to named volumes/folders** under `D:\docker-data` (or another dedicated partition) so backups and reinstalls are painless.
4. **Layer remote-first tooling:** Portainer (full-stack GUI), Yacht (one-click templates), and Watchtower (hands-free updates).
5. **Automate backups with Duplicati** to a cloud bucket and schedule monthly SMART tests plus UPS self-tests.
6. **Adopt an “immutable-infrastructure” mindset:** treat each service as a replaceable container; configs live in version-controlled `docker-compose` files.

If you follow the playbook exactly, you will be able to rebuild the entire server from bare metal in under an hour.

---

## 1 | Meet your hardware & objectives

### 1.1 — Why a mini PC makes a great home server

The X1 AI Pro ships with an AMD Ryzen AI 9 HX 370, 12 cores, 96 GB RAM and up to 4 TB NVMe storage—horsepower normally reserved for mid-range rack servers but in a silent, 15 W-65 W envelope .
**Goal:** run 15-30 lightweight services (media streaming, home-automation, DevOps toys, dashboards) on \<10 % average CPU and \<25 % RAM.

### 1.2 — Concept: container vs VM

A **virtual machine** emulates hardware and runs its own kernel; a **Docker container** shares the host kernel and isolates only the user-space layer. Containers start in milliseconds, use far less RAM, and snapshot as files called **images**.

---

## 2 | First-boot checklist (0 h – 1 h)

1. **Complete Windows OOBE** with a temporary local account.
2. **Plug into wired Ethernet**. Your server deserves a rock-solid link.
3. **Run Windows Update** until no patches remain. Reboot between cumulative updates.
4. **Flash the latest BIOS/UEFI** from Minisforum if offered (download on another PC, copy to USB). Always plug into UPS before flashing.
5. **Rename the PC** to something memorable (e.g., `x1-docker`)

```powershell
Rename-Computer -NewName "x1-docker" -Restart
```
6. **Set a DHCP reservation or static IP** so URLs like `https://x1-docker:9443` never change.
   *Control Panel → Network → Adapter settings → IPv4 → “Use the following IP”*.
7. **Turn on BitLocker** (TPM present) for the data drive.
8. **Create a dedicated admin user** (`docker-admin`), disable the default after testing.
9. **Install OpenSSH Server**—gives you headless shell access .

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
```
10. **Enable Windows features required for Docker.**

	```powershell
	dism /online /enable-feature /featurename:Microsoft-Hyper-V-All /all /norestart
	dism /online /enable-feature /featurename:VirtualMachinePlatform /norestart
	dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart
	```
11. **Reboot**.

*At this point the box is patched, virtualisation-ready, reachable by SSH, and safe to run headless.*

---

## 3 | Install Docker Desktop with WSL 2 (1 h – 1 h 30 m)

### Concept: WSL 2 backend

WSL 2 runs a real Linux kernel inside a lightweight VM managed by Windows. Docker Desktop can drive that kernel, so each Linux container behaves exactly as if you ran Docker on Ubuntu—no Hyper-V-style nested overhead .

#### Steps

1. **Install the latest WSL kernel**:

```powershell
wsl --install
wsl --set-default-version 2 :contentReference[oaicite:4]{index=4}
```
2. **Download Docker Desktop** (stable channel). Run the installer and **select “Use WSL 2 based engine”** when prompted.
3. **First-launch Docker Desktop** (Start Menu). It will ask permission to enable integration with your default WSL distro (usually `Ubuntu`). Accept.
4. **Settings → General**

   * Untick “Start Docker Desktop when you log in”,
   * Tick “Allow Docker to start on boot for Windows Server” (creates a system service) .
- **Settings → Resources → WSL Integration** — enable only the distros you plan to use (one is fine).
- **Verify**:

```powershell
docker version
docker run --rm hello-world
```

---

## 4 | Craft your directory & volume strategy (1 h 30 m – 2 h)

### Concept: named volumes vs bind-mounts

* **Named volume** is managed by Docker; lives under `\\wsl$\docker-desktop-data`.
* **Bind mount** points a container path to a real NTFS folder, giving you direct backup and indexing.

**Recommended layout** (on data drive `D:`):

```
D:\docker-data
│
├── compose      # docker-compose.yml files
├── volumes      # bind mounts
│   ├── portainer
│   ├── duplicati
│   ├── grafana
│   └── ...
└── backups      # duplicati target or cloud-sync cache
```

Create it:

```powershell
mkdir D:\docker-data\{compose,volumes,backups}
```

Point all stacks to these paths so you can snapshot or robocopy them.

---

## 5 | Deploy a remote management GUI (Portainer) (2 h – 2 h 15 m)

### Concept: Portainer

Portainer is a lightweight web UI that connects to your local Docker socket and lets you manage containers, images, networks and volumes graphically .

#### Steps

```powershell
docker volume create portainer_data
docker run -d `
  --name portainer `
  --restart=always `
  -p 8000:8000 -p 9443:9443 `
  -v /var/run/docker.sock:/var/run/docker.sock `
  -v portainer_data:/data `
  portainer/portainer-ce:latest
```

*Browse to* **`https://x1-docker:9443`** and create an admin password.

---

## 6 | Optional second UI: Yacht (template-driven) (2 h 15 m – 2 h 25 m)

### Concept: Yacht

Yacht focuses on *application templates*; one click spawns a pre-configured media server, database, etc. Useful if you prefer catalog-style installs .

```powershell
docker volume create yacht_data
docker run -d `
  --name yacht `
  --restart=always `
  -p 8001:8000 `
  -v /var/run/docker.sock:/var/run/docker.sock `
  -v yacht_data:/config `
  selfhostedpro/yacht:latest
```

Visit **`http://x1-docker:8001`**.

---

## 7 | Keep containers fresh automatically (Watchtower) (2 h 25 m – 2 h 30 m)

### Concept: Watchtower

Watchtower polls Docker Hub (or any registry) for newer images, pulls them, and recreates running containers with the original options—hands-free patching .

```powershell
docker run -d \
  --name watchtower \
  --restart=always \
  -e WATCHTOWER_CLEANUP=true \
  -e WATCHTOWER_SCHEDULE="0 0 4 * * *" `# 4 AM daily` \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower
```

*Tip:* Exclude critical databases by adding the container names to `--label-enable` or use `--scope`.

---

## 8 | Back up everything (Duplicati) (2 h 30 m – 3 h)

### Concept: Duplicati

Duplicati is an open-source backup daemon with a web UI; it compresses, encrypts and sends incremental backups to any cloud storage or local disk ([docs.duplicati.com][1], [duplicati.com][2]).

#### Steps

```powershell
docker run -d `
  --name duplicati `
  --restart=always `
  -p 8200:8200 `
  -v D:\docker-data\volumes\duplicati:/config `
  -v D:\docker-data:/source `
  -v D:\backups:/backups `
  duplicati/duplicati:latest
```

1. Visit **`http://x1-docker:8200`**, pick a strong passphrase.
2. Create a job:
   *Source* → `/source` (bind-mounted data folders)
   *Destination* → choose S3, OneDrive, Backblaze, etc.
   *Schedule* → nightly.
3. Test a restore once; document the steps.
   *Why not Windows File History?* Duplicati lives **inside** Docker, so you can port it to Linux later.

---

## 9 | Observe, alert, shut down safely (3 h – 3 h 30 m)

### 9.1 — System & container metrics with Netdata

Netdata provides 1-sec-resolution charts for CPU, RAM, disks, and every Docker container .

```powershell
docker run -d \
  --name netdata \
  --restart=always \
  -p 19999:19999 \
  -v netdataconfig:/etc/netdata \
  -v netdatalib:/var/lib/netdata \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --cap-add SYS_PTRACE \
  netdata/netdata:stable
```

Browse **`http://x1-docker:19999`**.

### 9.2 — Disk health: smartmontools

Install Windows-native smartctl to schedule weekly SMART tests and email alerts .

```powershell
choco install smartmontools -y
```

Add Task Scheduler job:

```
"C:\Program Files\smartmontools\bin\smartctl.exe" -a /dev/nvme0 > D:\logs\smart_nvme.txt
```

### 9.3 — UPS monitoring with Network UPS Tools (NUT)

Connect your UPS via USB and run the Windows NUT service so Docker shuts down gracefully during outages ([networkupstools.org][3], [networkupstools.org][4]).

* Quick installer: **WinNUT-Client** GUI ([github.com][5]).

---

## 10 | Security & certificates (3 h 30 m – 4 h)

1. **Enable Windows Defender real-time + reputation-based protection**.
2. **Firewall:** Docker publishes only the ports you map (`-p`). Add block-all inbound rule except those.
3. **TLS for dashboards:** use **win-acme** client to pull Let’s Encrypt certs even for internal domains via DNS challenge . Bind them in Portainer (`--sslcert`).
4. **SSH hardening:** edit `C:\ProgramData\ssh\sshd_config`

```
PasswordAuthentication no
AllowUsers brian docker-admin
```

   Restart service.

---

## 11 | Compose, Swarm & beyond (4 h – 4 h 40 m)

### 11.1 — Concept: Docker Compose v2

Compose is a YAML dialect that declares multi-container stacks; version 2 is built into the Docker CLI on Desktop .

*Example:*

```yaml
# D:\docker-data\compose\media.yml
services:
  jellyfin:
    image: jellyfin/jellyfin:latest
    ports:
      - "8096:8096"
    volumes:
      - D:\Media:/media
      - D:\docker-data\volumes\jellyfin:/config
    restart: unless-stopped
```

Launch: `docker compose -f D:\docker-data\compose\media.yml up -d`.

### 11.2 — Concept: Swarm

Docker Swarm mode turns multiple engines into one cluster. With only one node you gain little, but learning its declarative stacks is valuable.
`docker swarm init` then `docker stack deploy -c stack.yml mystack`.

### 11.3 — Concept: Kubernetes / k3d\*\*

If you later want full K8s, install **k3d** (runs k3s in Docker) on top of the same machine—Portainer can manage it.

---

## 12 | Headless operation tricks (4 h 40 m – 5 h)

* **Wake-on-LAN:** enable in BIOS and router so you can power on remotely.
* **Tailscale** or WireGuard: secure overlay network to access dashboards when away.
* **Chocolatey**: CLI package manager to update Windows software non-interactively.
* **SDelete / Storage Sense:** script monthly trim to keep NVMe fast.
* **Event Viewer custom views**: collect Docker-related service logs.

---

## 13 | Long-term maintenance & housekeeping (ongoing)

| Task                         | Frequency                            | Tool/Command                                 |
| ---------------------------- | ------------------------------------ | -------------------------------------------- |
| **Windows Updates**          | Patch Tuesday + zero-day out-of-band | `sconfig` or `winget upgrade --all --silent` |
| **Docker Engine update**     | Monthly                              | `winget upgrade --id Docker.DockerDesktop`   |
| **Container image pruning**  | Weekly                               | `docker system prune -af --volumes`          |
| **Watchtower check**         | Daily auto                           | Watchtower logs                              |
| **SMART short test**         | Weekly                               | `smartctl -t short /dev/nvme0`               |
| **SMART long test**          | Monthly                              | `smartctl -t long /dev/nvme0`                |
| **Duplicati backup verify**  | Monthly                              | Duplicati “Test restore”                     |
| **Netdata health alerts**    | Continuous                           | Email/Discord webhook                        |
| **UPS self-test**            | Quarterly                            | Manufacturer utility or `upsdrvctl`          |
| **Vacuum dust / clean fans** | Bi-annually                          | Physical                                     |

### Tips

* **Snapshot before big migrations.** Export stacks: `docker compose convert > stack.json`.
* **Use labels** (`com.example.backup=true`) to drive custom scripts.
* **Limit log size**: `--log-opt max-size=10m` on each container.
* **Document everything** in a Git repo: compose files, `.env`, backup keys.
* **Test disaster recovery once a year.** Restore onto a laptop or VM.

---

## 14 | Appendix: quick-reference commands

```powershell
# List running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Stop and remove everything (emergency reset)
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)

# Clean dangling images
docker image prune -f

# Export a named volume
docker run --rm -v volumename:/volume -v D:\backups:/backup alpine tar czf /backup/volumename.tar.gz /volume
```

---

### Final thoughts

The Minisforum X1 AI Pro gives you server-grade resources without the rack-mount hassle. By leaning on WSL 2 and Docker Desktop you sidestep dual-boot complexity, and with Portainer, Watchtower and Duplicati you gain the three pillars of modern homelab DevOps: **visibility, resiliency, and repeatability**. Keep the system patched, test your backups, and let your new headless companion quietly shoulder the workload while you experiment, learn and build. Happy self-hosting!

---

#### Citations

Docker Desktop & WSL 2  Hyper-V enablement  Portainer install guide  Watchtower GitHub readme  Yacht project site  Compose v2 migration notes  OpenSSH on Windows  win-acme reference  Netdata Docker monitoring  smartmontools project  Minisforum X1 AI Pro specs  Docker auto-start discussion  WSL command reference  Duplicati Docker docs ([docs.duplicati.com][6]) Duplicati product page ([duplicati.com][7]) Network UPS Tools site & docs ([networkupstools.org][8], [networkupstools.org][9])

[1]:	https://docs.duplicati.com/detailed-descriptions/using-duplicati-from-docker "Using Duplicati from Docker"
[2]:	https://duplicati.com/ "Duplicati"
[3]:	https://networkupstools.org/ "Network UPS Tools - Welcome"
[4]:	https://networkupstools.org/docs/user-manual.chunked/Overview.html "2. Network UPS Tools Overview"
[5]:	https://github.com/nutdotnet/WinNUT-Client "GitHub - nutdotnet/WinNUT-Client"
[6]:	https://docs.duplicati.com/detailed-descriptions/using-duplicati-from-docker "Using Duplicati from Docker"
[7]:	https://duplicati.com/ "Duplicati"
[8]:	https://networkupstools.org/ "Network UPS Tools - Welcome"
[9]:	https://networkupstools.org/docs/user-manual.chunked/Overview.html "2. Network UPS Tools Overview"