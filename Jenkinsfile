pipeline {
    agent any

    environment {
        // TODO: replace with your actual Artifactory Docker repo path once provisioned
        ARTIFACTORY_REGISTRY = '<your-artifactory-host>/<your-docker-repo>'
        IMAGE_NAME = 'cpai-service'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        // TODO: create this credential in Jenkins (Manage Jenkins > Credentials)
        // as a "Username with password" credential bound to your Artifactory account
        ARTIFACTORY_CREDENTIALS_ID = 'artifactory-credentials'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
            }
        }

        stage('Push to Artifactory') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: "${ARTIFACTORY_CREDENTIALS_ID}",
                    usernameVariable: 'ARTIFACTORY_USER',
                    passwordVariable: 'ARTIFACTORY_PASS'
                )]) {
                    sh '''
                        echo "$ARTIFACTORY_PASS" | docker login "$ARTIFACTORY_REGISTRY" -u "$ARTIFACTORY_USER" --password-stdin
                        docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ARTIFACTORY_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
                        docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ARTIFACTORY_REGISTRY/$IMAGE_NAME:latest"
                        docker push "$ARTIFACTORY_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
                        docker push "$ARTIFACTORY_REGISTRY/$IMAGE_NAME:latest"
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker logout "$ARTIFACTORY_REGISTRY" || true'
        }
    }
}
