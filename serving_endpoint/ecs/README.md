# How to use this Repo
## Warning
- Always make sure to destroy your API Service. Forgetting to do so could incur a large AWS fee
- Never commit your AWS Account ID to git. Save it in an `.env` file and ensure `.env` is added to your `.gitiginore`

## Setup, Deploy, and Destroy

### Setup Env Variables
Add an `.env` file containing your AWS account ID and region. Example file:
```
AWS_ACCOUNT_ID=1234567890
AWS_REGION=ap-southeast-1
```

Create a `backend.tf` file and add it to both `/infra/setup/backend.tf` and `/infra/app/backend.tf`. Example files:
```
terraform {
  backend "s3" {
    region = "<AWS_REGION>"
    bucket = "<BUCKET_NAME>"
    key    = "<APP_NAME>/terraform.tfstate"
  }
}
```
```
terraform {
  backend "s3" {
    region = "<AWS_REGION>"
    bucket = "<BUCKET_NAME>"
    key    = "<APP_NAME>/terraform.tfstate"
  }
}
```
Alternatively you can skip this step to store your Terraform state locally.

<br>

### Setup, Deploy, and Destroy Infrastructure/App
All of the following commands are run via the Makefile.

1. Setup your ECR Repository (one time)
    ```
    make setup-ecr
    ```

<br>

2. Build and deploy your container
    ```
    make deploy-container
    ```

<br>

3. Deploy your API Service on ECS Fargate
    ```
    make deploy-service
    ```
    Note: The URL for your endpoint will be printed by Terraform once the above command is done executing. Example: `alb_dns_name = "<APP_NAME>-alb-123456789.<AWS_REGION>.elb.amazonaws.com"`. Navigate to that URL in your browser to ensure the API is working. You can also check out the API docs at the `<URL>/docs` endpoint.

<br>

4. Destroy your API Service on ECS Fargate
    ```
    make destroy-service
    ```

Based on:
https://medium.com/aspiring-data-scientist/deploy-a-fastapi-app-on-aws-ecs-034b8b7b5ac2
https://github.com/tomsharp/fastapi-on-ecs/tree/main
