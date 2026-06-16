FROM ubuntu:22.04
ARG USE_GUROBI=0
ENV USE_GUROBI=${USE_GUROBI}
ARG MISSING_DATA=0
ARG GRB_VERSION=13.0.2
ENV GRB_VERSION=13.0.2
ARG GRB_SHORT_VERSION=13.0
ENV GRB_SHORT_VERSION=13.0
ARG TARGETPLATFORM
ARG PROJECT_DIR=/LASPATED
ENV GUROBI_HOME=/opt/gurobi/linux
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib"
COPY . $PROJECT_DIR

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    echo "armlinux64" > /platform.txt; \
    else \
    echo "linux64" > /platform.txt; \
    fi

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    gnupg \
    tzdata \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    libboost-all-dev \
    build-essential \
    vim \
    gdal-bin \
    libgdal-dev \
    p7zip-full \
    zip \
    wget \
    make \
    && update-ca-certificates \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.13 /tmp/get-pip.py \
    && python3.13 -m pip install --upgrade pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 100 \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip3.13 /usr/local/bin/pip \
    && ln -sf /usr/local/bin/pip3.13 /usr/local/bin/pip3 \
    && rm -f /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/*

RUN python3.13 -m pip install gurobipy==${GRB_VERSION}

RUN if [ "${USE_GUROBI}" = "1" ]; then \
    GRB_PLATFORM=$(cat /platform.txt) && \
    apt-get update && apt-get install --no-install-recommends -y ca-certificates wget && \
    update-ca-certificates && \
    wget -v https://packages.gurobi.com/${GRB_SHORT_VERSION}/gurobi${GRB_VERSION}_${GRB_PLATFORM}.tar.gz && \
    tar -xvf gurobi${GRB_VERSION}_${GRB_PLATFORM}.tar.gz && \
    rm -f gurobi${GRB_VERSION}_${GRB_PLATFORM}.tar.gz && \
    mv -f gurobi* gurobi && \
    rm -rf gurobi/${GRB_PLATFORM}/docs && \
    mv -f gurobi/${GRB_PLATFORM}* gurobi/linux; \
    fi

WORKDIR $PROJECT_DIR
RUN python3.13 -m pip install laspated scipy seaborn

RUN update-alternatives --set python3 /usr/bin/python3.13 \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3

RUN if [ "${USE_GUROBI}" = "1" ]; then \
    make -C Model_Calibration/Cpp USE_GUROBI=1 GUROBI_VER=130 && \
    make -C Model_Calibration/Cpp USE_GUROBI=1 GUROBI_VER=130 test; \
    else \
    make -C Model_Calibration/Cpp USE_GUROBI=0 && \
    make -C Model_Calibration/Cpp USE_GUROBI=0 test; \
    fi

RUN if [ "${USE_GUROBI}" = "1" ]; then \
    make -C Replication/cpp_tests USE_GUROBI=1 GUROBI_VER=130; \
    else \
    make -C Replication/cpp_tests USE_GUROBI=0; \
    fi

RUN if [ "${MISSING_DATA}" = "1" ]; then \
    g++ -o missing Missing_Data/Cpp/missing_data.cpp -DUSE_GUROBI=0 -std=c++14 -m64 -I../Model_Calibration/Cpp -lboost_program_options -O3; \
    fi
