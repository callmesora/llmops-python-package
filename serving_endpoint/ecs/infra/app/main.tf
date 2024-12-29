provider "aws" {
  region = var.region
  default_tags {
    tags = {
      app = var.app_name
    }
  }
}

module "network" {
  source   = "./network"
  app_name = var.app_name
  region   = var.region
}

module "ecs" {
  source             = "./ecs"
  app_name           = var.app_name
  region             = var.region
  image              = var.image
  vpc_id             = module.network.vpc.id
  public_subnet_ids  = [for s in module.network.public_subnets : s.id]
  private_subnet_ids = [for s in module.network.private_subnets : s.id]
  depends_on         = [module.network]
}


# Outputs
output "alb_dns_name" {
  value = module.ecs.alb_dns_name
}
