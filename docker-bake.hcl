target "sitl-gazebo" {
  context = "./sitl"
  dockerfile = "gazebo.Dockerfile"
  tags = [
    "ghcr.io/cpslab-asu/multicosim-greensight/sitl/gazebo:harmonic",
  ]
}

target "sitl-firmware" {
  context = "./sitl"
  dockerfile = "firmware.Dockerfile"
  tags = [
    "ghcr.io/cpslab-asu/multicosim-greensight/sitl/firmware:0.1.0",
  ]
}

target "sitl-imu" {
  context = "./sitl"
  dockerfile = "imu.Dockerfile"
  tags = [
    "ghcr.io/cpslab-asu/multicosim-greensight/sitl/imu:0.1.0",
  ]
}

group "sitl" {
  targets = ["sitl-gazebo", "sitl-firmware"]
}

group "imu" {
  targets = ["sitl", "sitl-imu"]
}
