/**
 * Tensor stream for expo-camera's CameraView + expo-gl (TensorFlow.js).
 * `cameraWithTensors` targets the old Expo Camera component; this mirrors that API using
 * CameraView's native ref for `createCameraTextureAsync`.
 */
import * as React from 'react';
import { PixelRatio, Platform, StyleSheet } from 'react-native';
import { CameraView } from 'expo-camera';
import { GLView } from 'expo-gl';
import {
  detectGLCapabilities,
  fromTexture,
  renderToGLView,
} from '@tensorflow/tfjs-react-native';

const DEFAULT_AUTORENDER = true;
const DEFAULT_RESIZE_DEPTH = 3;
const DEFAULT_USE_CUSTOM_SHADERS_TO_RESIZE = false;

function stripTensorProps(props) {
  const keys = new Set([
    'useCustomShadersToResize',
    'cameraTextureWidth',
    'cameraTextureHeight',
    'resizeWidth',
    'resizeHeight',
    'resizeDepth',
    'autorender',
    'onReady',
    'onStreamError',
    'rotation',
  ]);
  const out = {};
  Object.keys(props).forEach((key) => {
    if (!keys.has(key)) {
      out[key] = props[key];
    }
  });
  return out;
}

export class TensorCameraView extends React.Component {
  constructor(props) {
    super(props);
    this.onCameraLayout = this.onCameraLayout.bind(this);
    this.onGLContextCreate = this.onGLContextCreate.bind(this);
    this.state = { cameraLayout: null };
    this.camera = null;
    this.glView = null;
    this.glContext = null;
    this.rafID = null;
  }

  componentWillUnmount() {
    cancelAnimationFrame(this.rafID);
    const gl = this.glContext;
    this.glContext = null;
    if (gl) {
      GLView.destroyContextAsync(gl);
    }
    this.camera = null;
    this.glView = null;
  }

  onCameraLayout(event) {
    const { x, y, width, height } = event.nativeEvent.layout;
    this.setState({ cameraLayout: { x, y, width, height } });
  }

  /**
   * Native view passed to GL must be the inner Expo camera (CameraViewInterface), not the JS wrapper.
   */
  getNativeCameraTarget() {
    return this.camera?._cameraRef?.current ?? this.camera;
  }

  async createCameraTexture() {
    if (this.glView == null || this.camera == null) {
      throw new Error('Expo GL context or camera not available');
    }
    const delaysMs = [0, 120, 280, 450];
    let lastErr;
    for (const d of delaysMs) {
      if (d) {
        await new Promise((r) => setTimeout(r, d));
      }
      const nativeCam = this.getNativeCameraTarget();
      if (!nativeCam) {
        lastErr = new Error('Native camera view not ready');
        continue;
      }
      try {
        return await this.glView.createCameraTextureAsync(nativeCam);
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr ?? new Error('createCameraTextureAsync failed');
  }

  async onGLContextCreate(gl) {
    try {
      this.glContext = gl;
      const cameraTexture = await this.createCameraTexture();
      await detectGLCapabilities(gl);

      const autorender =
        this.props.autorender != null ? this.props.autorender : DEFAULT_AUTORENDER;
      const updatePreview = this.previewUpdateFunc(gl, cameraTexture);
      if (autorender) {
        const renderLoop = () => {
          updatePreview();
          gl.endFrameEXP();
          this.rafID = requestAnimationFrame(renderLoop);
        };
        renderLoop();
      }

      const { resizeDepth } = this.props;
      // Default camera texture dims match expo-camera + tfjs-react-native recommendations.
      // Zero dims can break fromTexture() on some devices.
      const cameraTextureHeight =
        this.props.cameraTextureHeight != null
          ? this.props.cameraTextureHeight
          : Platform.OS === 'ios'
            ? 1920
            : 1200;
      const cameraTextureWidth =
        this.props.cameraTextureWidth != null
          ? this.props.cameraTextureWidth
          : Platform.OS === 'ios'
            ? 1080
            : 1600;
      const useCustomShadersToResize =
        this.props.useCustomShadersToResize != null
          ? this.props.useCustomShadersToResize
          : DEFAULT_USE_CUSTOM_SHADERS_TO_RESIZE;

      const view = this;
      function* nextFrameGenerator() {
        const RGBA_DEPTH = 4;
        const textureDims = {
          height: cameraTextureHeight,
          width: cameraTextureWidth,
          depth: RGBA_DEPTH,
        };
        while (view.glContext != null) {
          const targetDims = {
            height: view.props.resizeHeight,
            width: view.props.resizeWidth,
            depth: resizeDepth || DEFAULT_RESIZE_DEPTH,
          };
          const imageTensor = fromTexture(
            gl,
            cameraTexture,
            textureDims,
            targetDims,
            useCustomShadersToResize,
            { rotation: view.props.rotation }
          );
          yield imageTensor;
        }
      }

      this.props.onReady(nextFrameGenerator(), updatePreview, gl, cameraTexture);
    } catch (err) {
      this.props.onStreamError?.(err);
    }
  }

  previewUpdateFunc(gl, cameraTexture) {
    const renderFunc = () => {
      const { cameraLayout } = this.state;
      if (!cameraLayout?.width || !cameraLayout?.height) {
        return;
      }
      const { rotation = 0 } = this.props;
      const width = PixelRatio.getPixelSizeForLayoutSize(cameraLayout.width);
      const height = PixelRatio.getPixelSizeForLayoutSize(cameraLayout.height);
      const isFrontCamera = this.camera?.props?.facing === 'front';
      const flipHorizontal = Platform.OS === 'ios' && isFrontCamera ? false : true;
      renderToGLView(gl, cameraTexture, { width, height }, flipHorizontal, rotation);
    };
    return renderFunc.bind(this);
  }

  render() {
    const { cameraLayout } = this.state;
    const cameraProps = stripTensorProps(this.props);
    const onlayout = this.props.onLayout
      ? (e) => {
          this.props.onLayout(e);
          this.onCameraLayout(e);
        }
      : this.onCameraLayout;
    cameraProps.onLayout = onlayout;

    const flatStyle = StyleSheet.flatten(this.props.style) || {};
    const baseZ = flatStyle.zIndex != null ? Number(flatStyle.zIndex) : 0;

    const cameraComp = (
      <CameraView
        key="tensor-camera-view"
        {...cameraProps}
        ref={(ref) => {
          this.camera = ref;
        }}
      />
    );

    let glViewComponent = null;
    if (cameraLayout != null) {
      const overlayStyles = StyleSheet.create({
        glView: {
          position: 'absolute',
          left: cameraLayout.x,
          top: cameraLayout.y,
          width: cameraLayout.width,
          height: cameraLayout.height,
          zIndex: baseZ + 1,
        },
      });
      glViewComponent = (
        <GLView
          key="tensor-gl-view"
          style={overlayStyles.glView}
          onContextCreate={this.onGLContextCreate}
          ref={(ref) => {
            this.glView = ref;
          }}
        />
      );
    }

    return [cameraComp, glViewComponent];
  }
}
