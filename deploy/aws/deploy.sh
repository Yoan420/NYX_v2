#!/usr/bin/env bash
# Run explicitly from an AWS-authorized workstation. Creates billable resources.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: AWS_REGION=eu-west-1 deploy/aws/deploy.sh \
  --stack-name nyx-forecasts --vpc-id vpc-... --subnet-ids subnet-...,subnet-... \
  [--instance-types m6i.xlarge,m6a.xlarge] [--schedule-state DISABLED|ENABLED]

Requires aws CLI v2, Docker BuildKit and permissions for the CloudFormation stack.
The schedule defaults to DISABLED. Upload a snapshot and test a manual job first.
IMAGE_TAG overrides the unique UTC build tag; ECR tags are immutable.
NYX_PRIVATE_REQUIREMENTS can point to private SDK requirements (BuildKit secret).
NYX_SATURN_WHEEL can point to the authorized tshistory_lite-0.5 Python wheel.
CODEX_PROXY_CERT, when set, is mounted for pip TLS trust during managed builds.
USAGE
}

stack_name=''
vpc_id=''
subnet_ids=''
instance_types='m6i.xlarge,m6a.xlarge'
schedule_state='DISABLED'
while (($#)); do
  case "$1" in
    --stack-name|--vpc-id|--subnet-ids|--instance-types|--schedule-state)
      if (($# < 2)); then usage >&2; exit 2; fi
      case "$1" in
        --stack-name) stack_name="$2" ;;
        --vpc-id) vpc_id="$2" ;;
        --subnet-ids) subnet_ids="$2" ;;
        --instance-types) instance_types="$2" ;;
        --schedule-state) schedule_state="$2" ;;
      esac
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ -z "$stack_name" || -z "$vpc_id" || -z "$subnet_ids" || -z "${AWS_REGION:-}" ]]; then
  usage >&2
  exit 2
fi
if [[ "$schedule_state" != DISABLED && "$schedule_state" != ENABLED ]]; then
  usage >&2
  exit 2
fi
command -v aws >/dev/null
command -v docker >/dev/null
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image_tag="${IMAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Bootstrap ECR with scheduling paused; a failed build cannot launch a missing image.
parameters=("VpcId=$vpc_id" "SubnetIds=$subnet_ids" "InstanceTypes=$instance_types" "ImageTag=$image_tag")
aws cloudformation deploy --region "$AWS_REGION" --stack-name "$stack_name" \
  --template-file "$repo_root/deploy/aws/batch.yaml" --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides "${parameters[@]}" ScheduleState=DISABLED
repository_uri="$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$stack_name" \
  --query "Stacks[0].Outputs[?OutputKey=='RepositoryUri'].OutputValue | [0]" --output text)"
if [[ -z "$repository_uri" || "$repository_uri" == None ]]; then
  echo 'ECR repository output missing.' >&2
  exit 1
fi
registry_host="${repository_uri%%/*}"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$registry_host"
build_options=(--platform linux/amd64 --tag "$repository_uri:$image_tag")
if [[ -n "${CODEX_PROXY_CERT:-}" ]]; then
  build_options+=(--secret "id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt")
fi
if [[ -n "${NYX_PRIVATE_REQUIREMENTS:-}" ]]; then
  build_options+=(--secret "id=private_requirements,src=$NYX_PRIVATE_REQUIREMENTS")
fi
if [[ -n "${NYX_SATURN_WHEEL:-}" ]]; then
  build_options+=(--secret "id=saturn_wheel,src=$NYX_SATURN_WHEEL")
fi
DOCKER_BUILDKIT=1 docker build "${build_options[@]}" "$repo_root"
docker push "$repository_uri:$image_tag"
aws cloudformation deploy --region "$AWS_REGION" --stack-name "$stack_name" \
  --template-file "$repo_root/deploy/aws/batch.yaml" --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides "${parameters[@]}" "ScheduleState=$schedule_state"
aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$stack_name" \
  --query 'Stacks[0].Outputs' --output table
