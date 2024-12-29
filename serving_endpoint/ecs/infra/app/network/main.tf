# Define provider
provider "aws" {
  region = var.region
  default_tags {
    tags = {
      app = var.app_name
    }
  }
}

# Create VPC and IGW
resource "aws_vpc" "this" {
  cidr_block = var.vpc_cidr_block
}
resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
}

# Create public subnets
resource "aws_subnet" "public_subnets" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.this.id
  cidr_block        = var.public_cidr_blocks[count.index]
  availability_zone = var.availability_zones[count.index]
}

# Create routing tables for public subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
}
resource "aws_route_table_association" "publics" {
  count          = length(var.availability_zones)
  subnet_id      = element(aws_subnet.public_subnets.*.id, count.index)
  route_table_id = aws_route_table.public.id
}


# Create Elastic IPs and NAT Gateways
resource "aws_eip" "eips" {
  count  = length(var.availability_zones)
  domain = "vpc"
}
resource "aws_nat_gateway" "this" {
  count         = length(var.availability_zones)
  subnet_id     = element(aws_subnet.public_subnets.*.id, count.index)
  allocation_id = element(aws_eip.eips.*.id, count.index)
}

# Create private subnets
resource "aws_subnet" "private_subnets" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.this.id
  cidr_block        = var.private_cidr_blocks[count.index]
  availability_zone = var.availability_zones[count.index]
}

# Create routing tables for private subnets
resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.this.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = element(aws_nat_gateway.this.*.id, count.index)
  }
}
resource "aws_route_table_association" "privates" {
  count          = length(var.availability_zones)
  subnet_id      = element(aws_subnet.private_subnets.*.id, count.index)
  route_table_id = element(aws_route_table.private.*.id, count.index)
}
