variable "app_name" {
  description = "Name of the app."
  type        = string
}
variable "region" {
  description = "AWS region to deploy the network to."
  type        = string
}
variable "image" {
  description = "Image used to start the container. Should be in repository-url/image:tag format."
  type        = string
}
