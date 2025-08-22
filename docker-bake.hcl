target "sitl-gazebo" {
  context = "./gazebo"
  tags = [
    "ghcr.io/cpslab-asu/multicosim-greensight/sitl/gazebo:harmonic",
  ]
}

group "sitl" {
  targets = ["sitl-gazebo"]
}
